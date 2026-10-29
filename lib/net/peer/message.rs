//! P2P message types

use std::{collections::HashSet, num::NonZeroUsize};

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use truthcoin_dc_types::Block;

use crate::{
    net::peer::{PeerState, PeerStateId},
    state::State,
    types::{AuthorizedTransaction, BlockHash, Header, Network, Tip, Txid},
};

pub const MAGIC_BYTES_LEN: usize = 4;

pub type MagicBytes = [u8; MAGIC_BYTES_LEN];

// First 25 bits are the US-TTY Baudot–Murray code for "TRUTH".
// The least significant bits of the 4th byte encode the network
// identifier.
pub const fn magic_bytes(network: Network) -> MagicBytes {
    const PREFIX: [u8; 4] =
        [0b1000_0010, 0b1000_1111, 0b0000_1010, 0b0000_0000];
    const fn network_identifier(network: Network) -> u8 {
        match network {
            Network::Regtest => 0b0000_0000,
            Network::Signet => 0b0000_0001,
            Network::Forknet => 0b0000_0010,
            // Network::Alphanet => 0b0000_0011,
            // Network::Betanet => 0b0000_0100,
            Network::Mainnet => 0b0000_0101,
        }
    }
    let mut res = PREFIX;
    *res.last_mut().unwrap() |= network_identifier(network);
    res
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct Heartbeat(pub PeerState);

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct GetBlockRequest {
    pub block_hash: BlockHash,
    /// Mainchain descendant tip that we are requesting the block to reach.
    /// Only relevant for the requester, so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub descendant_tip: Option<Tip>,
    /// Ancestor block. If no bodies are missing between `descendant_tip`
    /// and `ancestor`, then `descendant_tip` is ready to apply.
    /// Only relevant for the requester, so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub ancestor: Option<BlockHash>,
    /// Height of the requested block.
    /// Only relevant for the requester, so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub height: Option<u32>,
    /// Only relevant for the requester, so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub peer_state_id: Option<PeerStateId>,
}

impl GetBlockRequest {
    /// Upper bound of the bytes in a block response other than the body
    const BLOCK_RESPONSE_OVERHEAD: usize = 4096;

    /// Limit bytes to read in a response to a request
    pub fn read_response_limit(&self) -> NonZeroUsize {
        let height = self.height.expect(
            "GetBlock height should always be Some in an outbound request",
        );
        NonZeroUsize::new(
            State::body_size_limit(height) + Self::BLOCK_RESPONSE_OVERHEAD,
        )
        .unwrap()
    }
}

/// Request headers up to [`end`]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct GetHeadersRequest {
    /// Request headers AFTER (not including) the first ancestor found in
    /// the specified list, if such an ancestor exists.
    pub start: HashSet<BlockHash>,
    pub end: BlockHash,
    /// Height is only relevant for the requester,
    /// so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub height: Option<u32>,
    /// Only relevant for the requester, so serialization is skipped
    #[borsh(skip)]
    #[serde(skip)]
    pub peer_state_id: Option<PeerStateId>,
}

impl GetHeadersRequest {
    /// Hard cap on the bytes to read in a response, regardless of the
    /// requested height. Bounds memory use (and the response read timeout)
    /// even if an unreasonably large height is used to size the read budget.
    const MAX_READ_RESPONSE_LIMIT: NonZeroUsize =
        NonZeroUsize::new(64 * 1024 * 1024).unwrap();

    /// Limit bytes to read in a response to a request
    pub const fn read_response_limit(&self) -> NonZeroUsize {
        // 2KB limit per header
        const READ_HEADER_LIMIT: NonZeroUsize =
            NonZeroUsize::new(2048).unwrap();
        let expected_headers = (self.height.expect(
            "GetHeaders height should always be Some in an outbound request",
        ) as usize)
            .saturating_add(1);
        let limit = NonZeroUsize::new(expected_headers)
            .unwrap()
            .saturating_mul(READ_HEADER_LIMIT);
        if limit.get() > Self::MAX_READ_RESPONSE_LIMIT.get() {
            Self::MAX_READ_RESPONSE_LIMIT
        } else {
            limit
        }
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct PushTransactionRequest {
    pub transaction: AuthorizedTransaction,
}

impl PushTransactionRequest {
    /// Limit bytes to read in a response to a request
    pub const fn read_response_limit(&self) -> NonZeroUsize {
        // 256B limit per tx ack (response size is ~192)
        NonZeroUsize::new(256).unwrap()
    }
}

#[derive(BorshSerialize, Clone, Debug)]
pub enum Request {
    GetBlock(GetBlockRequest),
    GetHeaders(GetHeadersRequest),
    PushTransaction(PushTransactionRequest),
}

impl Request {
    /// Limit bytes to read in a response to a request
    pub fn read_response_limit(&self) -> NonZeroUsize {
        // TODO: Add constant for discriminant
        match self {
            Self::GetBlock(request) => request.read_response_limit(),
            Self::GetHeaders(request) => request.read_response_limit(),
            Self::PushTransaction(request) => request.read_response_limit(),
        }
    }
}

impl From<GetBlockRequest> for Request {
    fn from(request: GetBlockRequest) -> Self {
        Self::GetBlock(request)
    }
}

impl From<GetHeadersRequest> for Request {
    fn from(request: GetHeadersRequest) -> Self {
        Self::GetHeaders(request)
    }
}

impl From<PushTransactionRequest> for Request {
    fn from(request: PushTransactionRequest) -> Self {
        Self::PushTransaction(request)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RequestMessageRef<'a> {
    Heartbeat(&'a Heartbeat),
    Request(&'a Request),
}

impl RequestMessageRef<'_> {
    /// Limit bytes to read in a response to a request.
    /// Returns `None` if no response is expected
    pub fn read_response_limit(&self) -> Option<NonZeroUsize> {
        // TODO: Add constant for discriminant
        match self {
            Self::Heartbeat(_) => None,
            Self::Request(request) => Some(request.read_response_limit()),
        }
    }
}

impl<'a> From<&'a Heartbeat> for RequestMessageRef<'a> {
    fn from(heartbeat: &'a Heartbeat) -> Self {
        Self::Heartbeat(heartbeat)
    }
}

impl<'a> From<&'a Request> for RequestMessageRef<'a> {
    fn from(request: &'a Request) -> Self {
        Self::Request(request)
    }
}

impl<'a> Serialize for RequestMessageRef<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        enum Repr<'b> {
            Heartbeat(&'b Heartbeat),
            GetBlock(&'b GetBlockRequest),
            GetHeaders(&'b GetHeadersRequest),
            PushTransaction(&'b PushTransactionRequest),
        }

        let repr = match self {
            RequestMessageRef::Heartbeat(heartbeat) => {
                Repr::Heartbeat(heartbeat)
            }
            RequestMessageRef::Request(request) => match request {
                Request::GetBlock(request) => Repr::GetBlock(request),
                Request::GetHeaders(request) => Repr::GetHeaders(request),
                Request::PushTransaction(request) => {
                    Repr::PushTransaction(request)
                }
            },
        };
        repr.serialize(serializer)
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(transitive::Transitive, Debug)]
#[transitive(
    from(GetBlockRequest, Request),
    from(GetHeadersRequest, Request),
    from(PushTransactionRequest, Request)
)]
pub enum RequestMessage {
    Heartbeat(Heartbeat),
    Request(Request),
}

impl RequestMessage {
    pub fn as_ref<'a>(&'a self) -> RequestMessageRef<'a> {
        match self {
            Self::Heartbeat(heartbeat) => {
                RequestMessageRef::Heartbeat(heartbeat)
            }
            Self::Request(request) => RequestMessageRef::Request(request),
        }
    }
}

impl<'a> From<&'a RequestMessage> for RequestMessageRef<'a> {
    fn from(owned: &'a RequestMessage) -> Self {
        owned.as_ref()
    }
}

impl From<Heartbeat> for RequestMessage {
    fn from(heartbeat: Heartbeat) -> Self {
        Self::Heartbeat(heartbeat)
    }
}

impl From<Request> for RequestMessage {
    fn from(request: Request) -> Self {
        Self::Request(request)
    }
}

impl<'de> Deserialize<'de> for RequestMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Repr {
            Heartbeat(Heartbeat),
            GetBlock(GetBlockRequest),
            GetHeaders(GetHeadersRequest),
            PushTransaction(PushTransactionRequest),
        }
        let res = match Repr::deserialize(deserializer)? {
            Repr::Heartbeat(heartbeat) => heartbeat.into(),
            Repr::GetBlock(request) => request.into(),
            Repr::GetHeaders(request) => request.into(),
            Repr::PushTransaction(request) => request.into(),
        };
        Ok(res)
    }
}

#[derive(educe::Educe, Serialize, Deserialize)]
#[educe(Debug)]
pub enum ResponseMessage {
    Block(Box<Block>),
    /// Headers, from start to end
    Headers(#[educe(Debug(method(ResponseMessage::fmt_headers)))] Vec<Header>),
    NoBlock {
        block_hash: BlockHash,
    },
    NoHeader {
        block_hash: BlockHash,
    },
    TransactionAccepted(Txid),
    TransactionRejected(Txid),
}

impl ResponseMessage {
    /// Format headers for `Debug` impl
    fn fmt_headers(
        headers: &Vec<Header>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if let [first, .., last] = headers.as_slice() {
            write!(f, "[{first:?}, .., {last:?}]")
        } else {
            std::fmt::Debug::fmt(headers, f)
        }
    }
}

#[cfg(test)]
mod test {
    use bitcoin::hashes::Hash as _;
    use truthcoin_dc_types::{Block, Coinbase};

    use super::{GetBlockRequest, MagicBytes, ResponseMessage, magic_bytes};
    use crate::{
        state::State,
        types::{
            BlockHash, Body, FilledTransaction, Header, Network,
            UtreexoNodeHash,
        },
    };

    const NETWORKS: [Network; 4] = [
        Network::Forknet,
        Network::Mainnet,
        Network::Regtest,
        Network::Signet,
    ];

    fn thunder_magic_bytes(network: Network) -> MagicBytes {
        let id = match network {
            Network::Regtest => 0b0000_0000,
            Network::Signet => 0b0000_0001,
            Network::Forknet => 0b0000_0010,
            Network::Mainnet => 0b0000_0101,
        };
        [0b1000_0101, 0b0001_1000, 0b1001_0101, id]
    }

    #[test]
    fn magic_bytes_differ_from_thunder_on_every_network() {
        for ours in NETWORKS {
            for theirs in NETWORKS {
                assert_ne!(magic_bytes(ours), thunder_magic_bytes(theirs));
            }
        }
    }

    const BLOCKS_PER_MONTH: u32 = 6 * 24 * 30;

    #[test]
    fn largest_block_message_fits_read_cap() -> anyhow::Result<()> {
        let body = Body::new(Vec::new(), Coinbase::default());
        let header = Header {
            merkle_root: Body::compute_merkle_root::<FilledTransaction>(
                &body.coinbase,
                &[],
                &body.authorizations,
                &body.actor_proofs,
            )?,
            prev_side_hash: Some(BlockHash([1; 32])),
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            roots: vec![UtreexoNodeHash::Some([1; 32]); u64::BITS as usize],
        };
        let body_size = State::body_size(&body)?;
        let response = ResponseMessage::Block(Box::new(Block { header, body }));
        let overhead = bincode::serialized_size(&response)? - body_size;
        for month in 0..=121 {
            let height = month * BLOCKS_PER_MONTH;
            let request = GetBlockRequest {
                block_hash: BlockHash([0; 32]),
                descendant_tip: None,
                ancestor: None,
                height: Some(height),
                peer_state_id: None,
            };
            let largest = overhead + State::body_size_limit(height) as u64;
            let read_cap = request.read_response_limit().get() as u64;
            anyhow::ensure!(
                largest <= read_cap,
                "month {month}: block message of {largest} B exceeds the \
                 read cap of {read_cap} B",
            );
        }
        Ok(())
    }
}
