//! Task to communicate with mainchain node

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use bitcoin::{self, hashes::Hash as _};
use fallible_iterator::FallibleIterator;
use futures::{
    Stream, StreamExt as _, TryStreamExt as _,
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
};
use sneed::{EnvError, RwTxn, RwTxnError};
use tokio::{
    spawn,
    task::{self, JoinHandle},
};

use crate::{
    archive::{self, Archive},
    node::error::mainchain_task::{self as error, Error},
    types::{
        BmmResult, MainchainSyncPhase, MainchainSyncProgress,
        proto::{
            self,
            mainchain::{
                self, BlockHeaderInfo, Event as MainchainBlockEvent,
                ValidatorClient,
            },
        },
    },
    util::ErrorChain,
};

/// Request data from the mainchain node
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Request {
    /// Request missing mainchain ancestor header/infos
    AncestorInfos(bitcoin::BlockHash),
}

/// Response indicating that a request has been fulfilled
#[derive(Debug)]
pub(super) enum Response {
    /// Response bool indicates if the requested header was available
    AncestorInfos(bitcoin::BlockHash, Result<bool, error::Response>),
}

impl From<&Response> for Request {
    fn from(resp: &Response) -> Self {
        match resp {
            Response::AncestorInfos(block_hash, _) => {
                Request::AncestorInfos(*block_hash)
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum Event {
    Block(MainchainBlockEvent),
    Response(Response),
}

/// Progress of the startup sync, shared with the RPC server
#[derive(Clone, Default)]
struct SyncProgress(Arc<parking_lot::Mutex<MainchainSyncProgress>>);

impl SyncProgress {
    fn get(&self) -> MainchainSyncProgress {
        *self.0.lock()
    }

    /// `next_height` is the height of the next header to fetch
    fn fetched_headers(&self, tip_height: u32, next_height: Option<u32>) {
        *self.0.lock() = MainchainSyncProgress {
            phase: MainchainSyncPhase::Headers,
            done: tip_height - next_height.unwrap_or(0),
            total: tip_height,
            tip_height,
        };
    }

    fn start_writing(&self, tip_height: u32, total: u32) {
        *self.0.lock() = MainchainSyncProgress {
            phase: MainchainSyncPhase::Writing,
            done: 0,
            total,
            tip_height,
        };
    }

    fn wrote_headers(&self, headers: u32) {
        self.0.lock().done += headers;
    }

    fn start_state(&self, tip_height: u32, total: u32) {
        *self.0.lock() = MainchainSyncProgress {
            phase: MainchainSyncPhase::State,
            done: 0,
            total,
            tip_height,
        };
    }

    fn connected(&self, blocks: u32) {
        self.0.lock().done += blocks;
    }

    fn set_idle(&self) {
        *self.0.lock() = MainchainSyncProgress::default();
    }
}

struct ContextMut<'a, Transport> {
    env: &'a sneed::Env<heed::WithoutTls>,
    archive: &'a Archive,
    mainchain: &'a mut ValidatorClient<Transport>,
    sync_progress: &'a SyncProgress,
    event_tx: &'a mut UnboundedSender<Event>,
}

struct MainchainTask<Transport = tonic::transport::Channel> {
    env: sneed::Env<heed::WithoutTls>,
    archive: Archive,
    mainchain: ValidatorClient<Transport>,
    sync_progress: SyncProgress,
    // receive a request, and optional oneshot sender to send the result to
    // instead of sending on `response_tx`
    request_rx: UnboundedReceiver<(Request, Option<oneshot::Sender<Response>>)>,
    event_tx: UnboundedSender<Event>,
}

impl<Transport> MainchainTask<Transport>
where
    Transport: proto::Transport,
{
    /// Get best mainchain tip and subscribe to block events.
    /// There will be no missing events between the best mainchain tip and
    /// the first event in the returned stream.
    async fn subscribe_block_events(
        cusf_mainchain: &mut ValidatorClient<Transport>,
    ) -> Result<
        (
            bitcoin::BlockHash,
            impl Stream<Item = Result<MainchainBlockEvent, proto::Error>> + 'static,
        ),
        proto::Error,
    > {
        loop {
            tracing::debug!("attempting to subscribe to block events");
            let chain_tip = cusf_mainchain.get_chain_tip().await?.block_hash;
            let mut events = cusf_mainchain.subscribe_events().await?;
            tokio::select! {
                biased;

                _ = events.next() => {
                    continue
                }
                new_chain_tip = cusf_mainchain.get_chain_tip() => {
                    if chain_tip == new_chain_tip?.block_hash {
                        return Ok((chain_tip, events))
                    } else {
                        continue
                    }
                }
            }
        }
    }

    /// Request ancestor header info and block info from the mainchain node,
    /// including the specified header.
    /// Returns `false` if the specified block was not available.
    async fn request_ancestor_infos(
        env: &sneed::Env<heed::WithoutTls>,
        archive: &Archive,
        cusf_mainchain: &mut ValidatorClient<Transport>,
        sync_progress: &SyncProgress,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bool, error::RequestAncestorInfos> {
        if block_hash == bitcoin::BlockHash::all_zeros() {
            return Ok(true);
        } else {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            if archive
                .try_get_main_header_info(&rotxn, &block_hash)?
                .is_some()
            {
                return Ok(true);
            }
        }
        let mut current_block_hash = block_hash;
        let mut current_height = None;
        let mut block_infos =
            Vec::<(mainchain::BlockHeaderInfo, mainchain::BlockInfo)>::new();
        tracing::debug!(%block_hash, "requesting ancestor headers/info");
        const LOG_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
        const BATCH_REQUEST_SIZE: u32 = 20_000;
        let mut progress_logged = Instant::now();
        loop {
            if let Some(current_height) = current_height {
                let now = Instant::now();
                if now.duration_since(progress_logged) >= LOG_PROGRESS_INTERVAL
                {
                    progress_logged = now;
                    tracing::debug!(
                        %block_hash,
                        "requesting ancestor headers: {current_block_hash}({current_height} remaining)");
                }
                tracing::trace!(%block_hash, "requesting ancestor headers: {current_block_hash}({current_height})")
            }
            let Some(block_infos_resp) = cusf_mainchain
                .get_block_infos(current_block_hash, BATCH_REQUEST_SIZE - 1)
                .await?
            else {
                return Ok(false);
            };
            {
                let (current_header, _) = block_infos_resp.last();
                current_block_hash = current_header.prev_block_hash;
                current_height = current_header.height.checked_sub(1);
            }
            block_infos.extend(block_infos_resp);
            sync_progress
                .fetched_headers(block_infos[0].0.height, current_height);
            if current_block_hash == bitcoin::BlockHash::all_zeros() {
                break;
            } else {
                let rotxn = env.read_txn().map_err(EnvError::from)?;
                if archive
                    .try_get_main_header_info(&rotxn, &current_block_hash)?
                    .is_some()
                {
                    break;
                }
            }
        }
        let tip_height = block_infos[0].0.height;
        block_infos.reverse();
        // Writing all headers during IBD can starve archive readers.
        tracing::trace!(%block_hash, "storing ancestor headers/info");
        sync_progress.start_writing(tip_height, block_infos.len() as u32);
        task::block_in_place(|| {
            let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
            for (header_info, block_info) in block_infos {
                let () =
                    archive.put_main_header_info(&mut rwtxn, &header_info)?;
                let () = archive.put_main_block_info(
                    &mut rwtxn,
                    header_info.block_hash,
                    &block_info,
                )?;
                sync_progress.wrote_headers(1);
            }
            rwtxn.commit().map_err(RwTxnError::from)?;
            tracing::trace!(%block_hash, "stored ancestor headers/info");
            Ok(true)
        })
    }

    /// Sync mainchain state to the specified mainchain tip.
    /// Ancestor headers must exist for the specified mainchain tip.
    fn sync_side_tips_to_tip(
        mut rwtxn: RwTxn,
        archive: &Archive,
        sync_progress: &SyncProgress,
        mainchain_tip: bitcoin::BlockHash,
        event_tx: &mut UnboundedSender<Event>,
    ) -> Result<(), error::SyncSideTipsToTip> {
        let side_tips_tip = archive
            .side_tips()
            .get_mainchain_tip(&rwtxn)
            .map_err(archive::Error::from)?;
        let common_ancestor = if side_tips_tip.block_hash()
            == bitcoin::BlockHash::all_zeros()
            || mainchain_tip == bitcoin::BlockHash::all_zeros()
        {
            bitcoin::BlockHash::all_zeros()
        } else {
            archive.last_common_main_ancestor(
                &rwtxn,
                mainchain_tip,
                side_tips_tip.block_hash(),
            )?
        };
        let mut side_tips_tip_info = side_tips_tip.tip_info;
        let extract_tip = |tip_info: Option<BlockHeaderInfo>| {
            tip_info
                .map_or(bitcoin::BlockHash::all_zeros(), |info| info.block_hash)
        };
        let mut events = Vec::new();
        // disconnect mainchain state tip until common ancestor is reached
        while let Some(main_state_tip_info) = side_tips_tip_info {
            if main_state_tip_info.block_hash == common_ancestor {
                break;
            }
            let main_state_prev_tip_info = archive
                .main_parent_header_info(&rwtxn, &main_state_tip_info)?;
            let bmm_commitment = archive
                .get_main_block_info(&rwtxn, &main_state_tip_info.block_hash)?
                .bmm_commitment;
            let () = archive
                .side_tips()
                .disconnect_mainchain_tip(
                    &mut rwtxn,
                    main_state_prev_tip_info,
                    bmm_commitment,
                )
                .map_err(archive::Error::from)?;
            events.push(MainchainBlockEvent::DisconnectBlock {
                block_hash: main_state_tip_info.block_hash,
            });
            side_tips_tip_info = main_state_prev_tip_info;
        }
        if let Some(tip_info) =
            archive.try_get_main_header_info(&rwtxn, &mainchain_tip)?
        {
            let start_height =
                side_tips_tip_info.map_or(0, |info| info.height + 1);
            sync_progress.start_state(
                tip_info.height,
                tip_info.height + 1 - start_height,
            );
        }
        // connect mainchain state tip until mainchain tip is reached
        while extract_tip(side_tips_tip_info) != mainchain_tip {
            // Batch iterator items
            const BATCH_SIZE: usize = 32;
            let main_header_infos: smallvec::SmallVec<[_; BATCH_SIZE]> = {
                let start_height =
                    side_tips_tip_info.map_or(0, |info| info.height + 1);
                archive
                    .main_ancestor_header_infos_rev(
                        &rwtxn,
                        mainchain_tip,
                        start_height,
                    )
                    .take(BATCH_SIZE)
                    .collect()?
            };
            let batch_len = main_header_infos.len() as u32;
            for main_header_info in main_header_infos {
                let main_block_info = archive.get_main_block_info(
                    &rwtxn,
                    &main_header_info.block_hash,
                )?;
                let bmm_commitment = 'bmm_commitment: {
                    let Some(side_block_hash) = main_block_info.bmm_commitment
                    else {
                        break 'bmm_commitment None;
                    };
                    if !archive.contains_body(&rwtxn, &side_block_hash)? {
                        break 'bmm_commitment None;
                    }
                    let side_header =
                        archive.get_header(&rwtxn, side_block_hash)?;
                    if let Some(side_parent) = side_header.prev_side_hash
                        && !archive
                            .side_tips()
                            .sidechain_tips()
                            .contains_key(&rwtxn, &side_parent)
                            .map_err(archive::Error::from)?
                    {
                        break 'bmm_commitment None;
                    };
                    match archive.get_bmm_result(
                        &rwtxn,
                        side_block_hash,
                        main_header_info.block_hash,
                    )? {
                        BmmResult::Verified => {
                            Some(archive::side_tips::BmmCommitment {
                                sidechain_block_hash: side_block_hash,
                                sidechain_header_data: side_header.into(),
                            })
                        }
                        BmmResult::Failed => None,
                    }
                };
                let () = archive
                    .side_tips()
                    .connect_mainchain_tip(
                        &mut rwtxn,
                        main_header_info,
                        bmm_commitment,
                    )
                    .map_err(|err| {
                        let err =
                            archive::side_tips::Error::ConnectMainchainTip {
                                tip: main_header_info.block_hash,
                                source: err,
                            };
                        archive::Error::SideTips(err)
                    })?;
                side_tips_tip_info = Some(main_header_info);
                events.push(MainchainBlockEvent::ConnectBlock {
                    header_info: main_header_info,
                    block_info: main_block_info,
                });
            }
            sync_progress.connected(batch_len);
        }
        rwtxn.commit()?;
        // emit events
        for event in events {
            event_tx
                .unbounded_send(Event::Block(event))
                .map_err(|err| err.into_send_error())?;
        }
        Ok(())
    }

    fn handle_block_event(
        env: &sneed::Env<heed::WithoutTls>,
        archive: &Archive,
        event_tx: &mut UnboundedSender<Event>,
        event: proto::mainchain::Event,
    ) -> Result<(), error::HandleBlockEvent> {
        let mut rwtxn = env
            .write_txn()
            .map_err(|err| error::HandleBlockEvent::DbEnv(err.into()))?;
        match event {
            proto::mainchain::Event::ConnectBlock {
                header_info,
                ref block_info,
            } => {
                tracing::trace!(
                    block_hash = %header_info.block_hash,
                    "Handling connect block event"
                );
                let () =
                    archive.put_main_header_info(&mut rwtxn, &header_info)?;
                let () = archive.put_main_block_info(
                    &mut rwtxn,
                    header_info.block_hash,
                    block_info,
                )?;
                let bmm_commitment = 'bmm_commitment: {
                    let Some(side_block_hash) = block_info.bmm_commitment
                    else {
                        break 'bmm_commitment None;
                    };
                    if !archive.contains_body(&rwtxn, &side_block_hash)? {
                        break 'bmm_commitment None;
                    }
                    let side_header =
                        archive.get_header(&rwtxn, side_block_hash)?;
                    if let Some(side_parent) = side_header.prev_side_hash
                        && !archive
                            .side_tips()
                            .sidechain_tips()
                            .contains_key(&rwtxn, &side_parent)
                            .map_err(archive::Error::from)?
                    {
                        break 'bmm_commitment None;
                    };
                    match archive.get_bmm_result(
                        &rwtxn,
                        side_block_hash,
                        header_info.block_hash,
                    )? {
                        BmmResult::Verified => {
                            Some(archive::side_tips::BmmCommitment {
                                sidechain_block_hash: side_block_hash,
                                sidechain_header_data: side_header.into(),
                            })
                        }
                        BmmResult::Failed => None,
                    }
                };
                let () = archive
                    .side_tips()
                    .connect_mainchain_tip(
                        &mut rwtxn,
                        header_info,
                        bmm_commitment,
                    )
                    .map_err(|err| {
                        let err =
                            archive::side_tips::Error::ConnectMainchainTip {
                                tip: header_info.block_hash,
                                source: err,
                            };
                        archive::Error::SideTips(err)
                    })?;
            }
            proto::mainchain::Event::DisconnectBlock { block_hash } => {
                tracing::trace!(
                    %block_hash,
                    "Handling disconnect block event"
                );
                let header_info =
                    archive.get_main_header_info(&rwtxn, &block_hash)?;
                let bmm_commitment = archive
                    .get_main_block_info(&rwtxn, &block_hash)?
                    .bmm_commitment;
                let parent_info =
                    archive.main_parent_header_info(&rwtxn, &header_info)?;
                let () = archive
                    .side_tips()
                    .disconnect_mainchain_tip(
                        &mut rwtxn,
                        parent_info,
                        bmm_commitment,
                    )
                    .map_err(archive::Error::from)?;
            }
        }
        rwtxn
            .commit()
            .map_err(|err| error::HandleBlockEvent::DbWrite(err.into()))?;
        event_tx
            .unbounded_send(Event::Block(event))
            .map_err(|err| err.into_send_error())?;
        Ok(())
    }

    async fn handle_request(
        ctxt: ContextMut<'_, Transport>,
        request: Request,
        response_tx: Option<oneshot::Sender<Response>>,
    ) -> Result<(), Error> {
        match request {
            Request::AncestorInfos(main_block_hash) => {
                let res = Self::request_ancestor_infos(
                    ctxt.env,
                    ctxt.archive,
                    ctxt.mainchain,
                    ctxt.sync_progress,
                    main_block_hash,
                )
                .await
                .map_err(error::Response::from);
                ctxt.sync_progress.set_idle();
                let response = Response::AncestorInfos(main_block_hash, res);
                if let Some(response_tx) = response_tx {
                    response_tx
                        .send(response)
                        .map_err(|_: Response| Error::SendResponseOneshot)?;
                } else {
                    ctxt.event_tx
                        .unbounded_send(Event::Response(response))
                        .map_err(|err| {
                            Error::SendEvent(err.into_send_error())
                        })?;
                }
                Ok(())
            }
        }
    }

    async fn run_once(&mut self) -> Result<(), Error> {
        let (best_main_tip, block_event_stream) =
            Self::subscribe_block_events(&mut self.mainchain).await?;
        if !Self::request_ancestor_infos(
            &self.env,
            &self.archive,
            &mut self.mainchain,
            &self.sync_progress,
            best_main_tip,
        )
        .await
        .map_err(|err| Error::RequestAncestorInfos {
            tip: best_main_tip,
            source: err,
        })? {
            return Err(Error::AncestorInfoUnavailable { tip: best_main_tip });
        }
        {
            let rwtxn = self
                .env
                .write_txn()
                .map_err(|err| Error::DbEnv(err.into()))?;
            tracing::debug!(
                tip = %best_main_tip,
                "Syncing mainchain state to tip"
            );
            let () = Self::sync_side_tips_to_tip(
                rwtxn,
                &self.archive,
                &self.sync_progress,
                best_main_tip,
                &mut self.event_tx,
            )
            .map_err(|err| Error::SyncSideTipsToTip {
                tip: best_main_tip,
                source: Box::new(err),
            })?;
            self.sync_progress.set_idle();
        }
        enum MailboxItem {
            BlockEvent(proto::mainchain::Event),
            Request {
                request: Request,
                response_tx: Option<oneshot::Sender<Response>>,
            },
        }
        let block_event_stream =
            block_event_stream.map_ok(MailboxItem::BlockEvent);
        let request_stream =
            (&mut self.request_rx).map(|(request, response_tx)| {
                Ok(MailboxItem::Request {
                    request,
                    response_tx,
                })
            });
        let mut mailbox_stream =
            futures::stream::select(block_event_stream, request_stream);

        while let Some(item) = mailbox_stream.try_next().await? {
            match item {
                MailboxItem::BlockEvent(block_event) => {
                    let () = Self::handle_block_event(
                        &self.env,
                        &self.archive,
                        &mut self.event_tx,
                        block_event,
                    )?;
                }
                MailboxItem::Request {
                    request,
                    response_tx,
                } => {
                    let ctxt_mut = ContextMut {
                        env: &self.env,
                        archive: &self.archive,
                        mainchain: &mut self.mainchain,
                        sync_progress: &self.sync_progress,
                        event_tx: &mut self.event_tx,
                    };
                    let () =
                        Self::handle_request(ctxt_mut, request, response_tx)
                            .await?;
                }
            }
        }
        Ok(())
    }

    /// Run the task, and start it again after it stops. The mainchain node can
    /// stop at any time, and the node must connect to it again.
    async fn run(mut self) {
        const RECONNECT_DELAY: Duration = Duration::from_secs(5);

        loop {
            match self.run_once().await {
                Ok(()) => {
                    tracing::warn!("Mainchain task: the event stream closed")
                }
                Err(err) => tracing::error!(
                    "Mainchain task error: {:#}",
                    ErrorChain::new(&err)
                ),
            }
            self.sync_progress.set_idle();
            tokio::time::sleep(RECONNECT_DELAY).await;
            tracing::info!("Mainchain task: connecting to the mainchain node");
        }
    }
}

/// Handle to the task to communicate with mainchain node.
/// Task is aborted on drop.
#[derive(Clone)]
pub(super) struct MainchainTaskHandle {
    task: Arc<JoinHandle<()>>,
    sync_progress: SyncProgress,
    // send a request, and optional oneshot sender to receive the result on the
    // corresponding oneshot receiver
    request_tx:
        mpsc::UnboundedSender<(Request, Option<oneshot::Sender<Response>>)>,
}

impl MainchainTaskHandle {
    pub fn new<Transport>(
        env: sneed::Env<heed::WithoutTls>,
        archive: Archive,
        mainchain: ValidatorClient<Transport>,
    ) -> (Self, mpsc::UnboundedReceiver<Event>)
    where
        Transport: proto::Transport + Send + 'static,
        <Transport as tonic::client::GrpcService<tonic::body::Body>>::Future:
            Send,
    {
        let (request_tx, request_rx) = mpsc::unbounded();
        let (event_tx, event_rx) = mpsc::unbounded();
        let sync_progress = SyncProgress::default();
        let task = MainchainTask {
            env,
            archive,
            mainchain,
            sync_progress: sync_progress.clone(),
            request_rx,
            event_tx,
        };
        let task = spawn(task.run());
        let task_handle = MainchainTaskHandle {
            task: Arc::new(task),
            sync_progress,
            request_tx,
        };
        (task_handle, event_rx)
    }

    pub fn sync_progress(&self) -> MainchainSyncProgress {
        self.sync_progress.get()
    }

    /// Send a request
    pub fn request(&self, request: Request) -> Result<(), Request> {
        self.request_tx
            .unbounded_send((request, None))
            .map_err(|err| {
                let (request, _) = err.into_inner();
                request
            })
    }

    /// Send a request, and receive the response on a oneshot receiver instead
    /// of the response stream
    pub fn request_oneshot(
        &self,
        request: Request,
    ) -> Result<oneshot::Receiver<Response>, Request> {
        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        let () = self
            .request_tx
            .unbounded_send((request, Some(oneshot_tx)))
            .map_err(|err| {
                let (request, _) = err.into_inner();
                request
            })?;
        Ok(oneshot_rx)
    }
}

impl Drop for MainchainTaskHandle {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.task) {
            task.abort()
        }
    }
}

#[cfg(test)]
mod test {
    use std::{
        convert::Infallible,
        future::Ready,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll},
        time::Duration,
    };

    use bitcoin::hashes::Hash as _;
    use futures::channel::mpsc;
    use parking_lot::Mutex;
    use tonic::codegen::{BoxFuture, Service, http};

    use super::{MainchainTask, SyncProgress};
    use crate::{
        archive::{Archive, test::main_header_info},
        types::{
            MainchainSyncPhase, MainchainSyncProgress,
            proto::{
                common::{ConsensusHex, ReverseHex},
                mainchain::{ValidatorClient, generated},
            },
        },
    };

    fn temp_env()
    -> anyhow::Result<(temp_dir::TempDir, sneed::Env<heed::WithoutTls>)> {
        let temp_dir = temp_dir::TempDir::new()?;
        let mut opts = heed::EnvOpenOptions::new().read_txn_without_tls();
        opts.map_size(256 * 1024 * 1024).max_dbs(Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, temp_dir.path()) }?;
        Ok((temp_dir, env))
    }

    /// Serves `GetBlockInfo` for the chain of [`main_header_info`], and
    /// records the `max_ancestors` and the sync progress of each request
    #[derive(Clone, Default)]
    struct MockValidator {
        max_ancestors: Arc<Mutex<Vec<u32>>>,
        progress: Arc<Mutex<Vec<MainchainSyncProgress>>>,
        sync_progress: SyncProgress,
    }

    impl tonic::server::UnaryService<generated::GetBlockInfoRequest>
        for MockValidator
    {
        type Response = generated::GetBlockInfoResponse;
        type Future = Ready<
            Result<
                tonic::Response<generated::GetBlockInfoResponse>,
                tonic::Status,
            >,
        >;

        fn call(
            &mut self,
            request: tonic::Request<generated::GetBlockInfoRequest>,
        ) -> Ready<
            Result<
                tonic::Response<generated::GetBlockInfoResponse>,
                tonic::Status,
            >,
        > {
            let request = request.into_inner();
            let block_hash: bitcoin::BlockHash = request
                .block_hash
                .as_ref()
                .expect("block_hash")
                .decode::<generated::GetBlockInfoRequest, _>("block_hash")
                .expect("block_hash decodes");
            let max_ancestors = request.max_ancestors.expect("max_ancestors");
            self.max_ancestors.lock().push(max_ancestors);
            self.progress.lock().push(self.sync_progress.get());
            let height = u32::from_le_bytes(
                block_hash.as_byte_array()[1..5]
                    .try_into()
                    .expect("4 height bytes"),
            );
            let infos = (height.saturating_sub(max_ancestors)..=height)
                .rev()
                .map(|height| {
                    let info = main_header_info(height);
                    generated::get_block_info_response::Info {
                        header_info: Some(generated::BlockHeaderInfo {
                            block_hash: Some(ReverseHex::encode(
                                &info.block_hash,
                            )),
                            prev_block_hash: Some(ReverseHex::encode(
                                &info.prev_block_hash,
                            )),
                            height,
                            work: Some(ConsensusHex::encode(
                                &info.work.to_le_bytes(),
                            )),
                            timestamp: 0,
                        }),
                        block_info: Some(generated::BlockInfo::default()),
                    }
                })
                .collect();
            std::future::ready(Ok(tonic::Response::new(
                generated::GetBlockInfoResponse { infos },
            )))
        }
    }

    impl Service<http::Request<tonic::body::Body>> for MockValidator {
        type Response = http::Response<tonic::body::Body>;
        type Error = Infallible;
        type Future = BoxFuture<http::Response<tonic::body::Body>, Infallible>;

        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Infallible>> {
            Poll::Ready(Ok(()))
        }

        fn call(
            &mut self,
            request: http::Request<tonic::body::Body>,
        ) -> BoxFuture<http::Response<tonic::body::Body>, Infallible> {
            assert_eq!(
                request.uri().path(),
                "/cusf.mainchain.v1.ValidatorService/GetBlockInfo"
            );
            let mock = self.clone();
            Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(
                    tonic_prost::ProstCodec::default(),
                );
                Ok(grpc.unary(mock, request).await)
            })
        }
    }

    #[test]
    fn ancestor_walk_asks_for_20000_headers() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env()?;
        let archive = Archive::new(&env)?;
        let mock = MockValidator::default();
        let mut client = ValidatorClient::new(mock.clone());
        let tip = main_header_info(100).block_hash;
        let runtime = tokio::runtime::Runtime::new()?;
        let available = runtime.block_on(
            MainchainTask::<MockValidator>::request_ancestor_infos(
                &env,
                &archive,
                &mut client,
                &mock.sync_progress,
                tip,
            ),
        )?;
        assert!(available);
        assert_eq!(*mock.max_ancestors.lock(), [19_999]);
        Ok(())
    }

    #[test]
    fn sync_progress_walks_headers_writing_then_state() -> anyhow::Result<()> {
        const TIP_HEIGHT: u32 = 20_099;
        let (_temp_dir, env) = temp_env()?;
        let archive = Archive::new(&env)?;
        let mock = MockValidator::default();
        let sync_progress = &mock.sync_progress;
        let mut client = ValidatorClient::new(mock.clone());
        let tip = main_header_info(TIP_HEIGHT).block_hash;
        let stop_sampler = Arc::new(AtomicBool::new(false));
        let sampler = {
            let sync_progress = sync_progress.clone();
            let stop_sampler = stop_sampler.clone();
            std::thread::spawn(move || {
                let mut samples = Vec::new();
                while !stop_sampler.load(Ordering::Relaxed) {
                    samples.push(sync_progress.get());
                    std::thread::sleep(Duration::from_millis(1));
                }
                samples
            })
        };
        let runtime = tokio::runtime::Runtime::new()?;
        let available = runtime.block_on(
            MainchainTask::<MockValidator>::request_ancestor_infos(
                &env,
                &archive,
                &mut client,
                sync_progress,
                tip,
            ),
        )?;
        stop_sampler.store(true, Ordering::Relaxed);
        let samples = sampler.join().expect("the sampler thread ends");
        assert!(available);
        let headers = |done| MainchainSyncProgress {
            phase: MainchainSyncPhase::Headers,
            done,
            total: TIP_HEIGHT,
            tip_height: TIP_HEIGHT,
        };
        assert_eq!(
            *mock.progress.lock(),
            [MainchainSyncProgress::default(), headers(20_000)]
        );
        let written: Vec<u32> = samples
            .iter()
            .filter(|progress| {
                progress.phase == MainchainSyncPhase::Writing
                    && progress.total == TIP_HEIGHT + 1
                    && progress.tip_height == TIP_HEIGHT
            })
            .map(|progress| progress.done)
            .collect();
        assert!(
            written.iter().any(|done| *done > 0 && *done <= TIP_HEIGHT),
            "the write progress did not move: {written:?}"
        );
        assert_eq!(
            serde_json::to_value(sync_progress.get())?,
            serde_json::json!({
                "phase": "writing",
                "done": TIP_HEIGHT + 1,
                "total": TIP_HEIGHT + 1,
                "tip_height": TIP_HEIGHT,
            })
        );

        let (mut event_tx, _event_rx) = mpsc::unbounded();
        MainchainTask::<MockValidator>::sync_side_tips_to_tip(
            env.write_txn()?,
            &archive,
            sync_progress,
            tip,
            &mut event_tx,
        )?;
        assert_eq!(
            serde_json::to_value(sync_progress.get())?,
            serde_json::json!({
                "phase": "state",
                "done": TIP_HEIGHT + 1,
                "total": TIP_HEIGHT + 1,
                "tip_height": TIP_HEIGHT,
            })
        );

        sync_progress.set_idle();
        assert_eq!(
            serde_json::to_value(sync_progress.get())?,
            serde_json::json!({
                "phase": "idle",
                "done": 0,
                "total": 0,
                "tip_height": 0,
            })
        );
        Ok(())
    }
}
