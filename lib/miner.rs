use futures::TryStreamExt;

use crate::types::{
    Body, Header,
    proto::{self, mainchain},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error("merkle root mismatch: header and body are inconsistent")]
    MerkleRootMismatch,
    #[error("BMM candidate is stale; rebuild it against the current parent tip")]
    StaleCandidate,
}

#[derive(Clone)]
pub struct Miner<MainchainTransport = tonic::transport::Channel> {
    pub cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
    pub cusf_mainchain_wallet: mainchain::WalletClient<MainchainTransport>,
    block: Option<(Header, Body, u32)>,
}

impl<MainchainTransport> Miner<MainchainTransport> {
    pub fn new(
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        cusf_mainchain_wallet: mainchain::WalletClient<MainchainTransport>,
    ) -> Result<Self, Error> {
        Ok(Self {
            cusf_mainchain,
            cusf_mainchain_wallet,
            block: None,
        })
    }
}

impl<MainchainTransport> Miner<MainchainTransport>
where
    MainchainTransport: proto::Transport + Clone,
{
    pub async fn generate(&mut self) -> Result<(), Error> {
        let () = self.cusf_mainchain_wallet.generate_blocks(1).await?;
        Ok(())
    }

    pub async fn attempt_bmm(
        &mut self,
        amount: u64,
        height: u32,
        header: Header,
        body: Body,
    ) -> Result<bitcoin::Txid, Error> {
        if header.merkle_root != Body::compute_merkle_root(&body.coinbase, &body.transactions) {
            return Err(Error::MerkleRootMismatch);
        }
        let tip = self.cusf_mainchain.get_chain_tip().await?;
        if tip.block_hash != header.prev_main_hash || (height != 0 && height != tip.height) {
            return Err(Error::StaleCandidate);
        }
        let critical_hash = header.hash().0;
        let txid = self
            .cusf_mainchain_wallet
            .create_bmm_critical_data_tx(
                amount,
                tip.height,
                critical_hash,
                header.prev_main_hash,
            )
            .await?;
        tracing::info!(%txid, "created BMM tx");
        self.block = Some((header, body, tip.height));
        Ok(txid)
    }

    // Wait for a block to be connected that contains our BMM request.
    pub async fn confirm_bmm(
        &mut self,
    ) -> Result<Option<(bitcoin::BlockHash, Header, Body)>, Error> {
        use mainchain::Event;

        let Some((header, body, parent_height)) = self.block.take() else {
            return Ok(None);
        };
        // Subscribe before checking the tip so a block arriving during either
        // RPC cannot be missed. Recover an already-mined attempt from history.
        let mut events_client = self.cusf_mainchain.clone();
        let mut events = events_client.subscribe_events().await?;
        let tip = self.cusf_mainchain.get_chain_tip().await?;
        if tip.block_hash != header.prev_main_hash {
            let distance = tip.height.saturating_sub(parent_height);
            if distance > 0 && distance <= 10_000 {
                if let Some(infos) = self.cusf_mainchain.get_block_infos(tip.block_hash, distance - 1).await? {
                    for (info, block) in infos {
                        if info.prev_block_hash == header.prev_main_hash {
                            return Ok((block.bmm_commitment == Some(header.hash())).then_some((info.block_hash, header, body)));
                        }
                    }
                }
            }
            return Ok(None);
        }
        while let Some(event) = events.try_next().await? {
            match event {
                Event::ConnectBlock { header_info, block_info } => {
                    if header_info.prev_block_hash == header.prev_main_hash {
                        return Ok((block_info.bmm_commitment == Some(header.hash()))
                            .then_some((header_info.block_hash, header, body)));
                    }
                    // A different branch or an advanced tip requires a fresh
                    // candidate, with fresh deadline and execution validation.
                    if header_info.height > parent_height { return Ok(None); }
                }
                Event::DisconnectBlock { block_hash } => {
                    if block_hash == header.prev_main_hash { return Ok(None); }
                }
            }
        }
        Ok(None)
    }
}
