use std::net::{SocketAddrV4, SocketAddrV6};

use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, db, env};

use crate::{
    state::{
        error::Error,
        rollback::{RollBack, TxidStamped},
    },
    types::{EncryptionPubKey, Hash, VerifyingKey},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum VotecoinUpdate<T> {
    Delete,
    Retain,
    Set(T),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinDataUpdates {
    pub commitment: VotecoinUpdate<Hash>,
    pub socket_addr_v4: VotecoinUpdate<SocketAddrV4>,
    pub socket_addr_v6: VotecoinUpdate<SocketAddrV6>,
    pub encryption_pubkey: VotecoinUpdate<EncryptionPubKey>,
    pub signing_pubkey: VotecoinUpdate<VerifyingKey>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinNetworkData {
    pub commitment: Option<Hash>,
    pub socket_addr_v4: Option<SocketAddrV4>,
    pub socket_addr_v6: Option<SocketAddrV6>,
    pub encryption_pubkey: Option<EncryptionPubKey>,
    pub signing_pubkey: Option<VerifyingKey>,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct VotecoinId(pub Hash);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinData {
    pub(in crate::state) commitment: RollBack<TxidStamped<Option<Hash>>>,
    pub(in crate::state) socket_addr_v4:
        RollBack<TxidStamped<Option<SocketAddrV4>>>,
    pub(in crate::state) socket_addr_v6:
        RollBack<TxidStamped<Option<SocketAddrV6>>>,
    pub(in crate::state) encryption_pubkey:
        RollBack<TxidStamped<Option<EncryptionPubKey>>>,
    pub(in crate::state) signing_pubkey:
        RollBack<TxidStamped<Option<VerifyingKey>>>,
}

impl VotecoinData {
    pub fn at_block_height(&self, height: u32) -> Option<VotecoinNetworkData> {
        Some(VotecoinNetworkData {
            commitment: self.commitment.at_block_height(height)?.data,
            socket_addr_v4: self.socket_addr_v4.at_block_height(height)?.data,
            socket_addr_v6: self.socket_addr_v6.at_block_height(height)?.data,
            encryption_pubkey: self
                .encryption_pubkey
                .at_block_height(height)?
                .data,
            signing_pubkey: self.signing_pubkey.at_block_height(height)?.data,
        })
    }

    pub fn current(&self) -> VotecoinNetworkData {
        VotecoinNetworkData {
            commitment: self.commitment.latest().data,
            socket_addr_v4: self.socket_addr_v4.latest().data,
            socket_addr_v6: self.socket_addr_v6.latest().data,
            encryption_pubkey: self.encryption_pubkey.latest().data,
            signing_pubkey: self.signing_pubkey.latest().data,
        }
    }
}

#[derive(
    utoipa::ToSchema,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SeqId(pub u32);

#[derive(Clone)]
pub struct Dbs {
    votecoin:
        DatabaseUnique<SerdeBincode<VotecoinId>, SerdeBincode<VotecoinData>>,
    seq_to_votecoin:
        DatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<VotecoinId>>,
}

impl Dbs {
    pub const NUM_DBS: u32 = 2;

    pub(in crate::state) fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, env::error::CreateDb> {
        let votecoin = DatabaseUnique::create(env, rwtxn, "votecoin")?;
        let seq_to_votecoin =
            DatabaseUnique::create(env, rwtxn, "votecoin_seq_to_id")?;
        Ok(Self {
            seq_to_votecoin,
            votecoin,
        })
    }

    pub fn votecoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<VotecoinId>, SerdeBincode<VotecoinData>>
    {
        &self.votecoin
    }

    pub fn seq_to_votecoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<VotecoinId>> {
        &self.seq_to_votecoin
    }

    pub fn try_get_votecoin(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<Option<VotecoinData>, db::error::TryGet> {
        self.votecoin.try_get(rotxn, votecoin_id)
    }

    pub fn get_votecoin(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<VotecoinData, Error> {
        self.try_get_votecoin(rotxn, votecoin_id)?.ok_or(
            Error::UnbalancedVotecoin {
                inputs: 0,
                outputs: 0,
            },
        )
    }

    pub fn try_get_votecoin_data_at_block_height(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
        height: u32,
    ) -> Result<Option<VotecoinNetworkData>, db::error::TryGet> {
        let res = self
            .votecoin
            .try_get(rotxn, votecoin_id)?
            .and_then(|votecoin_data| votecoin_data.at_block_height(height));
        Ok(res)
    }

    pub fn try_get_current_votecoin_data(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<Option<VotecoinNetworkData>, Error> {
        let res = self
            .votecoin
            .try_get(rotxn, votecoin_id)?
            .map(|votecoin_data| votecoin_data.current());
        Ok(res)
    }
}
