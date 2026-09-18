use std::path::PathBuf;

use sneed::{
    EnvError, RwTxnError,
    db::{self, error::Error as DbError},
    env, rwtxn,
};

use crate::{
    archive::side_tips::error as side_tips,
    types::{BlockHash, Txid, Version},
};

#[allow(clippy::duplicated_attributes)]
#[derive(thiserror::Error, transitive::Transitive, Debug)]
#[transitive(from(db::error::Delete, DbError))]
#[transitive(from(db::error::Get, DbError))]
#[transitive(from(db::error::Last, DbError))]
#[transitive(from(db::error::Put, DbError))]
#[transitive(from(db::error::TryGet, DbError))]
#[transitive(from(env::error::CreateDb, EnvError))]
#[transitive(from(env::error::WriteTxn, EnvError))]
#[transitive(from(rwtxn::error::Commit, RwTxnError))]
#[transitive(from(side_tips::DisconnectMainchainTip, side_tips::Error))]
#[transitive(from(side_tips::DisconnectSidechainTip, side_tips::Error))]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error(
        "Incompatible DB version ({}). Please clear the DB (`{}`) and re-sync",
        .version,
        .db_path.display()
    )]
    IncompatibleVersion { version: Version, db_path: PathBuf },
    #[error("invalid previous side hash")]
    InvalidPrevSideHash,
    #[error("invalid merkle root")]
    InvalidMerkleRoot,
    #[error("no ancestor with depth {depth} for block {block_hash}")]
    NoAncestor { block_hash: BlockHash, depth: u32 },
    #[error("no block with hash {0}")]
    NoBlock(BlockHash),
    #[error("unknown block hash: {0}")]
    NoBlockHash(BlockHash),
    #[error("no BMM result with block {0}")]
    NoBmmResult(BlockHash),
    #[error("no block body with hash {0}")]
    NoBody(BlockHash),
    #[error("no deposits info for block {0}")]
    NoDepositsInfo(bitcoin::BlockHash),
    #[error("no header with hash {0}")]
    NoHeader(BlockHash),
    #[error("no height info for block hash {0}")]
    NoHeight(BlockHash),
    #[error("no mainchain ancestor with depth {depth} for block {block_hash}")]
    NoMainAncestor {
        block_hash: bitcoin::BlockHash,
        depth: u32,
    },
    #[error("unknown mainchain block hash: {0}")]
    NoMainBlockHash(bitcoin::BlockHash),
    #[error("no mainchain block info for block hash {0}")]
    NoMainBlockInfo(bitcoin::BlockHash),
    #[error("no mainchain header info for block hash {0}")]
    NoMainHeaderInfo(bitcoin::BlockHash),
    #[error("no height info for mainchain block hash {0}")]
    NoMainHeight(bitcoin::BlockHash),
    #[error("no tx with txid {0}")]
    NoTx(Txid),
    #[error(transparent)]
    SideTips(#[from] side_tips::Error),
}
