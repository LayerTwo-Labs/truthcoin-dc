use std::{
    collections::{HashMap, HashSet, hash_map},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use fallible_iterator::FallibleIterator;
use futures::{StreamExt, channel::mpsc};
use heed::types::{SerdeBincode, Unit};
use hickory_resolver::TokioResolver;
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use sneed::{
    DatabaseUnique, Env, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey,
    db::error::Error as DbError,
};
use tokio_stream::StreamNotifyClose;
use tracing::instrument;
use truthcoin_dc_types::authorization::BatchVerificationContext;

use crate::{
    archive::Archive,
    state::State,
    types::{
        AuthorizedTransaction, Network, VERSION, Version,
        net::{
            DEFAULT_PORT, Peer, PeerAddress, PeerConnectionStatus,
            ResolvedPeerAddress,
        },
    },
    util::ErrorChain,
};

pub mod error;
mod peer;

pub use error::Error;
use peer::{
    Connection, ConnectionContext as PeerConnectionCtxt,
    ConnectionHandle as PeerConnectionHandle,
};
pub use peer::{
    ConnectionError as PeerConnectionError, Info as PeerConnectionInfo,
    InternalMessage as PeerConnectionMessage, PeerStateId,
    Request as PeerRequest, ResponseMessage as PeerResponse,
    message as peer_message,
};

/// Dummy certificate verifier that treats any certificate as valid.
/// NOTE, such verification is vulnerable to MITM attacks, but convenient for testing.
#[derive(Debug)]
struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer,
        _intermediates: &[rustls::pki_types::CertificateDer],
        _server_name: &rustls::pki_types::ServerName,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn configure_client() -> Result<ClientConfig, error::ConfigureClient> {
    let crypto_provider = Arc::new(rustls::crypto::ring::default_provider());
    let crypto = rustls::ClientConfig::builder_with_provider(crypto_provider)
        .with_safe_default_protocol_versions()
        .map_err(error::configure_client::Inner::Rustls)?
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    let client_config =
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
    Ok(ClientConfig::new(Arc::new(client_config)))
}

/// Returns default server configuration along with its certificate.
fn configure_server(
    mut server_names: HashSet<String>,
) -> Result<(ServerConfig, Vec<u8>), Error> {
    server_names.insert("localhost".to_owned());
    let server_names = Vec::from_iter(server_names);
    let cert_key = rcgen::generate_simple_self_signed(server_names)?;
    let keypair_der = cert_key.key_pair.serialize_der();
    let priv_key = rustls::pki_types::PrivateKeyDer::Pkcs8(keypair_der.into());
    let cert_der = cert_key.cert.der().to_vec();
    let cert_chain = vec![cert_key.cert.into()];

    let mut server_config =
        ServerConfig::with_single_cert(cert_chain, priv_key)?;
    let transport_config = Arc::get_mut(&mut server_config.transport).unwrap();
    transport_config.max_concurrent_uni_streams(1_u8.into());

    Ok((server_config, cert_der))
}

/// Constructs a QUIC endpoint configured to listen for incoming connections on a certain address
/// and port.
///
/// ## Returns
///
/// - a stream of incoming QUIC connections
/// - server certificate serialized into DER format
pub fn make_server_endpoint(
    bind_addr: SocketAddr,
    server_names: HashSet<String>,
) -> Result<(Endpoint, Vec<u8>), Error> {
    let (server_config, server_cert) = configure_server(server_names)?;
    tracing::info!("creating server endpoint: binding to {bind_addr}",);
    let mut endpoint =
        Endpoint::server(server_config, bind_addr).map_err(Error::Quinn)?;
    let client_cfg = configure_client()?;
    endpoint.set_default_client_config(client_cfg);
    Ok((endpoint, server_cert))
}

// None indicates that the stream has ended
pub type PeerInfoRx =
    mpsc::UnboundedReceiver<(SocketAddr, Option<PeerConnectionInfo>)>;

const BETANET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    const ECASH_EU_COM: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("seed.beta.ecash.eu.com"),
        port: DEFAULT_PORT,
    };
    &[ECASH_EU_COM]
};

const SIGNET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    const SIGNET_MINING_SERVER: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(172, 105, 148, 135)),
        port: DEFAULT_PORT,
    };
    const BIP300_XYZ: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("truthcoin.bip300.xyz"),
        port: DEFAULT_PORT,
    };
    &[SIGNET_MINING_SERVER, BIP300_XYZ]
};

const FORKNET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    const BIP300_XYZ: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("explorer.bip300.xyz"),
        port: DEFAULT_PORT,
    };
    &[
        BIP300_XYZ,
        PeerAddress {
            host: url::Host::Ipv4(Ipv4Addr::new(157, 180, 8, 224)),
            port: DEFAULT_PORT,
        },
    ]
};

/// Add every seed address the network names that the database does not hold.
/// A datadir made before a seed existed would otherwise never learn it.
fn ensure_seed_peers(
    known_peers: &DatabaseUnique<SerdeBincode<PeerAddress>, Unit>,
    rwtxn: &mut RwTxn,
    network: Network,
) -> Result<(), DbError> {
    for seed_peer_addr in seed_peer_addrs(network) {
        let seed_peer_addr = PeerAddress::to_owned(seed_peer_addr);
        if known_peers.try_get(rwtxn, &seed_peer_addr)?.is_none() {
            known_peers.put(rwtxn, &seed_peer_addr, &())?;
        }
    }
    Ok(())
}

const fn seed_peer_addrs(
    network: Network,
) -> &'static [PeerAddress<&'static str>] {
    match network {
        Network::Betanet => BETANET_SEED_PEER_ADDRS,
        Network::Forknet => FORKNET_SEED_PEER_ADDRS,
        Network::Regtest => &[],
        Network::Signet => SIGNET_SEED_PEER_ADDRS,
    }
}

pub async fn resolve_peer_address<S>(
    dns_resolver: &TokioResolver,
    peer_addr: PeerAddress<S>,
) -> Result<ResolvedPeerAddress<S>, error::ResolvePeerAddress>
where
    S: std::fmt::Display,
    for<'a> &'a S: hickory_resolver::proto::rr::IntoName,
{
    match peer_addr.host {
        url::Host::Ipv4(ipv4) => Ok(ResolvedPeerAddress::Static(
            SocketAddr::new(IpAddr::V4(ipv4), peer_addr.port),
        )),
        url::Host::Ipv6(ipv6) => Ok(ResolvedPeerAddress::Static(
            SocketAddr::new(IpAddr::V6(ipv6), peer_addr.port),
        )),
        url::Host::Domain(domain) => {
            let mut addrs: Vec<_> = dns_resolver
                .lookup_ip(&domain)
                .await
                .map_err(|err| error::ResolvePeerAddress::Net(Box::new(err)))?
                .into_iter()
                .filter(|addr| !addr.is_unspecified())
                .collect();
            if let Some(last_addr) = addrs.pop() {
                addrs.reverse();
                let addrs = nonempty::NonEmpty {
                    head: last_addr,
                    tail: addrs,
                };
                Ok(ResolvedPeerAddress::Domain {
                    port: peer_addr.port,
                    addrs,
                    domain,
                })
            } else {
                let domain = domain.to_string();
                tracing::warn!(%domain, "unable to resolve host");
                Err(error::ResolvePeerAddress::NoIpAddrs { domain })
            }
        }
    }
}

/// Handle to tasks that dial known peers. Tasks are aborted on drop.
#[repr(transparent)]
pub struct DialKnownPeersHandle(
    tokio_util::task::JoinMap<PeerAddress, Result<bool, error::DialKnownPeer>>,
);

impl DialKnownPeersHandle {
    pub async fn join_next(
        &mut self,
    ) -> Option<(
        PeerAddress,
        Result<Result<bool, error::DialKnownPeer>, tokio::task::JoinError>,
    )> {
        self.0.join_next().await
    }
}

// Keep track of peer state
// Exchange metadata
// Bulk download
// Propagation
//
// Initial block download
//
// 1. Download headers
// 2. Download blocks
// 3. Update the state
#[derive(Clone)]
pub struct Net {
    pub server: Endpoint,
    archive: Archive,
    pub(crate) batch_verification_ctxt: BatchVerificationContext,
    pub dns_resolver: Arc<TokioResolver>,
    magic_bytes: peer_message::MagicBytes,
    state: State,
    active_peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectionHandle>>>,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: DatabaseUnique<SerdeBincode<PeerAddress>, Unit>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Net {
    pub const NUM_DBS: u32 = 2;

    fn add_active_peer(
        &self,
        addr: SocketAddr,
        peer_connection_handle: PeerConnectionHandle,
        info_rx: mpsc::UnboundedReceiver<PeerConnectionInfo>,
    ) -> Result<(), error::AlreadyConnected> {
        tracing::trace!(%addr, "add active peer: starting");
        let mut active_peers_write = self.active_peers.write();
        match active_peers_write.entry(addr) {
            hash_map::Entry::Occupied(_) => {
                tracing::error!(%addr, "add active peer: already connected");
                return Err(error::AlreadyConnected(addr));
            }
            hash_map::Entry::Vacant(active_peer_entry) => {
                active_peer_entry.insert(peer_connection_handle);
            }
        }
        drop(active_peers_write);
        tokio::spawn({
            let info_rx = StreamNotifyClose::new(info_rx)
                .map(move |info| Ok((addr, info)));
            let peer_info_tx = self.peer_info_tx.clone();
            async move {
                if let Err(_send_err) = info_rx.forward(peer_info_tx).await {
                    tracing::error!(%addr, "Failed to send peer connection info");
                }
            }
        });
        Ok(())
    }

    pub fn remove_active_peer(&self, addr: SocketAddr) {
        tracing::trace!(%addr, "remove active peer: starting");
        let mut active_peers_write = self.active_peers.write();
        if let Some(peer_connection) = active_peers_write.remove(&addr) {
            drop(peer_connection);
            tracing::info!(%addr, "remove active peer: disconnected");
        }
    }

    /// Apply the provided function to the peer connection handle,
    /// if it exists.
    pub fn try_with_active_peer_connection<F, T>(
        &self,
        addr: SocketAddr,
        f: F,
    ) -> Option<T>
    where
        F: FnMut(&PeerConnectionHandle) -> T,
    {
        let active_peers_read = self.active_peers.read();
        active_peers_read.get(&addr).map(f)
    }

    // TODO: This should have more context.
    // Last received message, connection state, etc.
    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.active_peers
            .read()
            .iter()
            .map(|(addr, conn_handle)| Peer {
                address: *addr,
                status: conn_handle.connection_status(),
            })
            .collect()
    }

    #[instrument(skip_all, fields(addr), err(Debug))]
    pub fn connect_peer(
        &self,
        env: Env<heed::WithoutTls>,
        mut resolved_addr: ResolvedPeerAddress,
    ) -> Result<(), error::ConnectPeer> {
        {
            let mut rwtxn = env.write_txn()?;
            self.known_peers.put(
                &mut rwtxn,
                &resolved_addr.as_peer_address().to_owned(),
                &(),
            )?;
            rwtxn.commit()?;
        }
        {
            let active_peers = self.active_peers.read();
            for ip_addr in resolved_addr.ip_addrs() {
                let addr = SocketAddr::new(ip_addr, resolved_addr.port());
                if active_peers.contains_key(&addr) {
                    tracing::error!("already connected to peer");
                    return Err(error::AlreadyConnected(addr).into());
                }
            }
        }
        let (addr, connecting) = loop {
            let addr = SocketAddr::new(
                resolved_addr.first_ip_addr(),
                resolved_addr.port(),
            );
            // Quinn makes this check too, but its error only says "invalid
            // remote address". This one names the address.
            if addr.ip().is_unspecified() {
                return Err(error::ConnectPeer::UnspecfiedPeerIP(addr.ip()));
            }
            let server_name = match resolved_addr.host() {
                url::Host::Domain(domain) => domain.as_str(),
                url::Host::Ipv4(_) | url::Host::Ipv6(_) => "localhost",
            };
            match self.server.connect(addr, server_name) {
                Ok(connecting) => break (addr, connecting),
                Err(err @ quinn::ConnectError::InvalidRemoteAddress(_)) => {
                    let (_, Some(next_addr)) =
                        resolved_addr.pop_first_ip_addr()
                    else {
                        return Err(err.into());
                    };
                    resolved_addr = next_addr;
                }
                Err(err) => return Err(err.into()),
            }
        };
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            batch_verification_ctxt: self.batch_verification_ctxt,
            magic_bytes: self.magic_bytes,
            resolved_address: resolved_addr,
            state: self.state.clone(),
        };

        let (connection_handle, info_rx) =
            peer::connect(connecting, connection_ctxt);
        self.add_active_peer(addr, connection_handle, info_rx)?;
        Ok(())
    }

    /// Delete peer from known_peers DB.
    /// Connections to the peer are not terminated.
    pub fn forget_peer(
        &self,
        rwtxn: &mut RwTxn,
        peer_address: &PeerAddress,
    ) -> Result<bool, Error> {
        self.known_peers
            .delete(rwtxn, peer_address)
            .map_err(|err| DbError::from(err).into())
    }

    fn known_peer_addrs(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<PeerAddress>, DbError> {
        let peer_addrs = self.known_peers.iter_keys(rotxn)?.collect()?;
        Ok(peer_addrs)
    }

    fn is_active_peer(&self, resolved_addr: &ResolvedPeerAddress) -> bool {
        let active_peers = self.active_peers.read();
        resolved_addr.ip_addrs().any(|ip_addr| {
            active_peers
                .contains_key(&SocketAddr::new(ip_addr, resolved_addr.port()))
        })
    }

    /// Dial a peer that the database knows.
    /// Returns `true` if a connection started, and `false` if the peer is
    /// already connected.
    async fn dial_known_peer(
        &self,
        env: Env<heed::WithoutTls>,
        peer_addr: PeerAddress,
    ) -> Result<bool, error::DialKnownPeer> {
        tracing::trace!("connecting to already known peer at {peer_addr}");
        let resolved_peer_addr =
            resolve_peer_address(&self.dns_resolver, peer_addr)
                .await
                .map_err(error::DialKnownPeer::DnsResolve)?;
        if self.is_active_peer(&resolved_peer_addr) {
            return Ok(false);
        }
        let () = self.connect_peer(env, resolved_peer_addr)?;
        Ok(true)
    }

    /// Dial every peer that the database knows, seeds included.
    /// Returns the number of connections that started.
    async fn dial_known_peers(
        &self,
        env: &Env<heed::WithoutTls>,
    ) -> Result<usize, Error> {
        let peer_addrs = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            self.known_peer_addrs(&rotxn)?
        };
        let mut dialed = 0;
        for peer_addr in peer_addrs {
            match self
                .dial_known_peer(Env::clone(env), peer_addr.clone())
                .await
            {
                Ok(true) => dialed += 1,
                Ok(false) => (),
                Err(err) => {
                    tracing::error!(%peer_addr, message = %ErrorChain::new(&err))
                }
            }
        }
        Ok(dialed)
    }

    /// Dial the known peers again while no peer connection exists.
    /// `min_delay` is the shortest wait between two checks for a connection.
    /// The wait doubles after each redial, up to `max_delay`.
    /// The future returns only on a database error.
    pub async fn redial_known_peers(
        &self,
        env: Env<heed::WithoutTls>,
        min_delay: Duration,
        max_delay: Duration,
    ) -> Result<(), Error> {
        let mut delay = min_delay;
        let mut no_peers_at_last_check = false;
        loop {
            tokio::time::sleep(delay).await;
            let active_peer_count = self.active_peers.read().len();
            if active_peer_count != 0 {
                delay = min_delay;
                no_peers_at_last_check = false;
                continue;
            }
            // The net task reconnects to a peer that errored. A redial waits
            // for a full delay with no connection, so it never dials first.
            if !no_peers_at_last_check {
                no_peers_at_last_check = true;
                continue;
            }
            let dialed = self.dial_known_peers(&env).await?;
            tracing::info!(dialed, "no peer connection: dialed known peers");
            delay = (2 * delay).min(max_delay);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &tokio::runtime::Handle,
        env: &Env<heed::WithoutTls>,
        archive: Archive,
        batch_verification_ctxt: BatchVerificationContext,
        magic_bytes_override: Option<peer_message::MagicBytes>,
        network: Network,
        state: State,
        bind_addr: SocketAddr,
        add_peers: HashSet<PeerAddress>,
        server_names: HashSet<String>,
    ) -> Result<(Self, PeerInfoRx, DialKnownPeersHandle), Error> {
        let (server, _) = make_server_endpoint(bind_addr, server_names)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        let known_peers =
            match DatabaseUnique::open(env, &rwtxn, "known_peers")? {
                Some(known_peers) => known_peers,
                None => DatabaseUnique::create(env, &mut rwtxn, "known_peers")?,
            };
        let () = ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
        for peer in add_peers {
            known_peers.put(&mut rwtxn, &peer, &())?;
        }
        let version = DatabaseUnique::create(env, &mut rwtxn, "net_version")?;
        match version.try_get(&rwtxn, &())? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 17,
                        patch: 6,
                    } =>
            {
                // types for `known_peers` db changed in v0.17.6
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version.put(&mut rwtxn, &(), &*VERSION)?,
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        let magic_bytes = magic_bytes_override
            .unwrap_or_else(|| peer_message::magic_bytes(network));
        let dns_resolver = {
            let builder = hickory_resolver::Resolver::builder_tokio()
                .map_err(Error::BuildDnsResolver)?;
            let resolver = builder.build().map_err(Error::BuildDnsResolver)?;
            Arc::new(resolver)
        };
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let net = Net {
            server,
            archive,
            batch_verification_ctxt,
            dns_resolver,
            magic_bytes,
            state,
            active_peers,
            peer_info_tx,
            known_peers: known_peers.clone(),
            _version: version,
        };
        let known_peers = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            net.known_peer_addrs(&rotxn)?
        };
        let dial_known_peers_handle = {
            let mut join_map = tokio_util::task::JoinMap::new();
            for peer_addr in known_peers {
                let env = Env::clone(env);
                let net = net.clone();
                join_map.spawn_on(
                    peer_addr.clone(),
                    async move {
                        net.dial_known_peer(env, peer_addr)
                            .await
                            .inspect_err(|err| tracing::error!(message = %ErrorChain::new(err)))
                    },
                    runtime
                );
            }
            DialKnownPeersHandle(join_map)
        };
        Ok((net, peer_info_rx, dial_known_peers_handle))
    }

    /// Accept the next incoming connection. Returns Some(addr) if a connection was accepted
    /// and a new peer was added.
    pub async fn accept_incoming(
        &self,
        env: Env<heed::WithoutTls>,
    ) -> Result<Option<SocketAddr>, error::AcceptConnection> {
        tracing::debug!(
            "accept incoming: listening for connections on `{}`",
            self.server
                .local_addr()
                .map(|socket| socket.to_string())
                .unwrap_or("unknown address".into())
        );
        let connection = match self.server.accept().await {
            Some(conn) => {
                let remote_address = conn.remote_address();
                tracing::trace!("accepting connection from {remote_address}",);

                let raw_conn = conn.await.map_err(|error| {
                    error::AcceptConnection::Connection {
                        error,
                        remote_address,
                    }
                })?;
                Connection::new(raw_conn, self.magic_bytes)
            }
            None => {
                tracing::debug!("server endpoint closed");
                return Err(error::AcceptConnection::ServerEndpointClosed);
            }
        };
        let addr = connection.addr();

        tracing::trace!(%addr, "accepted incoming connection");
        if self.active_peers.read().contains_key(&addr) {
            tracing::info!(
                %addr, "incoming connection: already peered, refusing duplicate",
            );
            connection
                .inner
                .close(quinn::VarInt::from_u32(1), b"already connected");
        }
        if connection.inner.close_reason().is_some() {
            return Ok(None);
        }
        tracing::info!(%addr, "connected to new peer");
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            batch_verification_ctxt: self.batch_verification_ctxt,
            magic_bytes: self.magic_bytes,
            resolved_address: addr.into(),
            state: self.state.clone(),
        };
        let (connection_handle, info_rx) =
            peer::handle(connection_ctxt, connection);
        self.add_active_peer(addr, connection_handle, info_rx)?;
        Ok(Some(addr))
    }

    /// Attempt to push an internal message to the specified peer
    /// Returns `true` if successful
    pub fn push_internal_message(
        &self,
        message: PeerConnectionMessage,
        addr: SocketAddr,
    ) -> bool {
        let active_peers_read = self.active_peers.read();
        let Some(peer_connection_handle) = active_peers_read.get(&addr) else {
            let err = Error::MissingPeerConnection(addr);
            tracing::warn!("{:#}", ErrorChain::new(&err));
            return false;
        };

        if let Err(send_err) = peer_connection_handle
            .internal_message_tx
            .unbounded_send(message)
        {
            let message = send_err.into_inner();
            tracing::warn!(
                "Failed to push internal message to peer connection {addr}: {message:?}"
            );
            return false;
        }
        true
    }

    /// Push a tx to all active peers, except those in the provided set
    pub fn push_tx(
        &self,
        exclude: HashSet<SocketAddr>,
        tx: &AuthorizedTransaction,
    ) {
        self.active_peers
            .read()
            .iter()
            .filter(|(addr, _)| !exclude.contains(addr))
            .for_each(|(addr, peer_connection_handle)| {
                match peer_connection_handle.connection_status() {
                    PeerConnectionStatus::Connecting => {
                        tracing::trace!(%addr, "skipping peer at {addr} because it is not fully connected");
                        return;
                    }
                    PeerConnectionStatus::Connected => {}
                }
                let request: PeerRequest = peer::message::PushTransactionRequest {
                    transaction: tx.clone(),
                }.into();
                if let Err(_send_err) = peer_connection_handle
                    .internal_message_tx
                    .unbounded_send(request.into())
                {
                    let txid = tx.transaction.txid();
                    tracing::warn!("Failed to push tx {txid} to peer at {addr}")
                }
            })
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::HashSet,
        net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
        time::Duration,
    };
    use truthcoin_dc_types::authorization::BatchVerificationContext;

    use anyhow::Context;
    use futures::StreamExt;

    use heed::types::{SerdeBincode, Unit};
    use sneed::DatabaseUnique;

    use crate::{
        archive::Archive,
        net::{
            Net, PeerAddress, PeerConnectionInfo, PeerInfoRx,
            ensure_seed_peers, make_server_endpoint, resolve_peer_address,
            seed_peer_addrs,
        },
        state::State,
        types::{Network, net::ResolvedPeerAddress},
    };

    fn temp_env(
        test_name: &str,
    ) -> anyhow::Result<(temp_dir::TempDir, sneed::Env<heed::WithoutTls>)> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "truthcoin-{test_name}-{}-{nanos}",
            std::process::id()
        ))?;
        let mut opts = heed::EnvOpenOptions::new().read_txn_without_tls();
        opts.map_size(16 * 1024 * 1024)
            .max_dbs(Archive::NUM_DBS + State::NUM_DBS + Net::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, temp_dir.path()) }?;
        Ok((temp_dir, env))
    }

    fn temp_net(
        test_name: &str,
    ) -> anyhow::Result<(
        temp_dir::TempDir,
        sneed::Env<heed::WithoutTls>,
        Net,
        PeerInfoRx,
    )> {
        let (temp_dir, env) = temp_env(test_name)?;
        let archive = Archive::new(&env)?;
        let state = State::new(&env, None)?;
        let (net, info_rx, _known_peers) = Net::new(
            &tokio::runtime::Handle::current(),
            &env,
            archive,
            BatchVerificationContext::new(&mut rand::rng()),
            None,
            Network::Regtest,
            state,
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
            HashSet::new(),
        )?;
        Ok((temp_dir, env, net, info_rx))
    }

    #[tokio::test]
    async fn rejected_duplicate_has_no_peer_close_event() -> anyhow::Result<()>
    {
        let (_temp_dir, env, net, info_rx) = temp_net("peer-duplicate")?;
        let (remote, _) = make_server_endpoint(
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
        )?;
        let addr = remote.local_addr()?;
        net.connect_peer(env.clone(), addr.into())?;
        let connection_ctxt = super::PeerConnectionCtxt {
            env,
            archive: net.archive.clone(),
            batch_verification_ctxt: net.batch_verification_ctxt,
            magic_bytes: net.magic_bytes,
            resolved_address: addr.into(),
            state: net.state.clone(),
        };
        let (duplicate, duplicate_info) = super::peer::connect(
            net.server.connect(addr, "localhost")?,
            connection_ctxt,
        );

        let error = net
            .add_active_peer(addr, duplicate, duplicate_info)
            .unwrap_err();
        assert_eq!(error.0, addr);
        assert_eq!(net.get_active_peers().len(), 1);
        drop(net);

        let events = tokio::time::timeout(
            Duration::from_secs(5),
            info_rx.collect::<Vec<_>>(),
        )
        .await?;
        assert_eq!(events.iter().filter(|(_, info)| info.is_none()).count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn connect_peer_skips_ipv6_on_an_ipv4_endpoint() -> anyhow::Result<()>
    {
        let (_temp_dir, env, net, mut info_rx) = temp_net("peer-family")?;
        let (remote, _) = make_server_endpoint(
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
        )?;
        let addr = remote.local_addr()?;
        let next_ip = Ipv4Addr::new(127, 0, 0, 2);
        let resolved = ResolvedPeerAddress::Domain {
            domain: "localhost".to_owned(),
            port: addr.port(),
            addrs: nonempty::NonEmpty {
                head: next_ip.into(),
                tail: vec![
                    Ipv4Addr::LOCALHOST.into(),
                    Ipv6Addr::LOCALHOST.into(),
                ],
            },
        };

        net.connect_peer(env, resolved)?;

        let peers = net.get_active_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].address, addr);
        assert!(net.server.local_addr()?.is_ipv4());
        net.server.close(0_u32.into(), b"test complete");
        let (reported_addr, info) =
            tokio::time::timeout(Duration::from_secs(5), info_rx.next())
                .await?
                .context("the peer task returned no result")?;
        let Some(PeerConnectionInfo::Error {
            resolved_peer_addr, ..
        }) = info
        else {
            anyhow::bail!("the peer task returned no connection error");
        };
        assert_eq!(reported_addr, addr);
        assert_eq!(
            resolved_peer_addr.ip_addrs().collect::<Vec<_>>(),
            vec![IpAddr::V4(Ipv4Addr::LOCALHOST), next_ip.into()]
        );
        Ok(())
    }

    #[tokio::test]
    async fn connect_peer_returns_the_last_invalid_address()
    -> anyhow::Result<()> {
        let (_temp_dir, env, net, _info_rx) = temp_net("peer-ipv6")?;
        let addr = SocketAddr::from((Ipv6Addr::LOCALHOST, 4009));
        let resolved = ResolvedPeerAddress::Domain {
            domain: "localhost".to_owned(),
            port: addr.port(),
            addrs: nonempty::NonEmpty {
                head: addr.ip(),
                tail: vec!["::2".parse()?],
            },
        };

        let error = net.connect_peer(env, resolved).unwrap_err();

        assert!(matches!(
            error,
            super::error::ConnectPeer::QuinnConnect(
                quinn::ConnectError::InvalidRemoteAddress(failed)
            ) if failed == addr
        ));
        assert!(net.get_active_peers().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn connect_peer_keeps_a_static_ipv4_address() -> anyhow::Result<()> {
        let (_temp_dir, env, net, _info_rx) = temp_net("peer-static")?;
        let (remote, _) = make_server_endpoint(
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
        )?;
        let addr = remote.local_addr()?;

        net.connect_peer(env, addr.into())?;

        let peers = net.get_active_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].address, addr);
        Ok(())
    }

    #[tokio::test]
    async fn connect_peer_returns_other_quinn_errors() -> anyhow::Result<()> {
        let (_temp_dir, env, net, _info_rx) = temp_net("peer-closed")?;
        net.server.close(0_u32.into(), b"test complete");
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 4009));

        let error = net.connect_peer(env, addr.into()).unwrap_err();

        assert!(matches!(
            error,
            super::error::ConnectPeer::QuinnConnect(
                quinn::ConnectError::EndpointStopping
            )
        ));
        assert!(net.get_active_peers().is_empty());
        Ok(())
    }

    const TEST_REDIAL_MIN_DELAY: Duration = Duration::from_millis(50);
    const TEST_REDIAL_MAX_DELAY: Duration = Duration::from_millis(200);

    /// A peer that drops leaves no connection, so the node dials it again.
    #[tokio::test]
    async fn a_lost_peer_is_dialed_again() -> anyhow::Result<()> {
        let (_temp_dir, env, net, _info_rx) = temp_net("peer-redial")?;
        let (remote, _) = make_server_endpoint(
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
        )?;
        let addr = remote.local_addr()?;
        net.connect_peer(env.clone(), addr.into())?;
        assert_eq!(net.get_active_peers().len(), 1);
        net.remove_active_peer(addr);
        assert!(net.get_active_peers().is_empty());
        let redial = tokio::spawn({
            let env = env.clone();
            let net = net.clone();
            async move {
                net.redial_known_peers(
                    env,
                    TEST_REDIAL_MIN_DELAY,
                    TEST_REDIAL_MAX_DELAY,
                )
                .await
            }
        });

        let dialed_again =
            tokio::time::timeout(Duration::from_secs(5), async {
                while net.get_active_peers().is_empty() {
                    tokio::time::sleep(TEST_REDIAL_MIN_DELAY).await;
                }
            })
            .await;

        redial.abort();
        dialed_again.context("the node dialed the lost peer no more")?;
        assert_eq!(net.get_active_peers()[0].address, addr);
        Ok(())
    }

    /// A peer that holds a connection takes no redial, and the loop starts no
    /// second connection to it.
    #[tokio::test]
    async fn a_connected_peer_takes_no_redial() -> anyhow::Result<()> {
        let (_temp_dir, env, net, _info_rx) = temp_net("peer-redial-skip")?;
        let (remote, _) = make_server_endpoint(
            (Ipv4Addr::LOCALHOST, 0).into(),
            HashSet::new(),
        )?;
        let addr = remote.local_addr()?;
        net.connect_peer(env.clone(), addr.into())?;
        let peer_addr = PeerAddress {
            host: url::Host::Ipv4(Ipv4Addr::LOCALHOST),
            port: addr.port(),
        };

        assert!(!net.dial_known_peer(env.clone(), peer_addr).await?);
        assert_eq!(net.dial_known_peers(&env).await?, 0);

        let redial = tokio::spawn({
            let env = env.clone();
            let net = net.clone();
            async move {
                net.redial_known_peers(
                    env,
                    TEST_REDIAL_MIN_DELAY,
                    TEST_REDIAL_MAX_DELAY,
                )
                .await
            }
        });
        tokio::time::sleep(TEST_REDIAL_MAX_DELAY * 5).await;
        redial.abort();
        let peers = net.get_active_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].address, addr);
        Ok(())
    }

    /// Every seed reaches a peer table that already exists, and a second call
    /// writes the same set.
    #[test]
    fn seeds_reach_an_existing_database() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("seed-peers")?;
        let network = Network::Betanet;
        let known_peers = {
            let mut rwtxn = env.write_txn()?;
            let known_peers: DatabaseUnique<SerdeBincode<PeerAddress>, Unit> =
                DatabaseUnique::create(&env, &mut rwtxn, "known_peers")?;
            ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
            ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
            rwtxn.commit()?;
            known_peers
        };
        let rotxn = env.read_txn()?;
        for seed_peer_addr in seed_peer_addrs(network) {
            let seed_peer_addr = PeerAddress::to_owned(seed_peer_addr);
            anyhow::ensure!(
                known_peers.try_get(&rotxn, &seed_peer_addr)?.is_some(),
                "the seed {seed_peer_addr} never reached the database"
            );
        }
        assert_eq!(
            known_peers.len(&rotxn)?,
            seed_peer_addrs(network).len() as u64
        );
        Ok(())
    }

    /// A seed names a host and a port, and the resolver keeps both.
    #[tokio::test]
    async fn a_seed_name_resolves_with_its_port() -> anyhow::Result<()> {
        let dns_resolver =
            hickory_resolver::Resolver::builder_tokio()?.build()?;
        let peer_addr: PeerAddress = "localhost:4009".parse()?;
        let resolved = resolve_peer_address(&dns_resolver, peer_addr).await?;
        assert_eq!(resolved.port(), 4009);
        assert!(
            resolved
                .ip_addrs()
                .any(|addr| addr == IpAddr::V4(Ipv4Addr::LOCALHOST)
                    || addr == IpAddr::V6(Ipv6Addr::LOCALHOST))
        );
        Ok(())
    }
}
