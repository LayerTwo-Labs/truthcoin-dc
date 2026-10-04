use thiserror::Error;

#[derive(Debug, Error)]
#[error("Bitcoin amount overflow")]
pub struct AmountOverflow;

#[derive(Debug, Error)]
#[error("Bitcoin amount underflow")]
pub struct AmountUnderflow;

#[derive(Debug, Error)]
pub enum Bech32mDecode {
    #[error(transparent)]
    Bech32m(#[from] bech32::DecodeError),
    #[error(
        "Wrong Bech32 HRP. Perhaps this key is being used somewhere it shouldn't be."
    )]
    WrongHrp,
    #[error("Wrong decoded byte length. Must decode to 32 bytes of data.")]
    WrongSize,
    #[error("Wrong Bech32 variant. Only Bech32m is accepted.")]
    WrongVariant,
}

#[derive(Debug, Error)]
#[error("invalid decision ID: {reason}")]
pub struct InvalidDecisionId {
    pub reason: String,
}

#[derive(Debug, Error)]
#[error("body has fewer authorizations than transaction inputs")]
pub struct MalformedBody;

pub mod withdrawal_bundle {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub(crate) enum Inner {
        #[error(
            "bundle too heavy: weight `{weight}` > max weight `{max_weight}`"
        )]
        BundleTooHeavy { weight: u64, max_weight: u64 },
    }

    #[derive(Debug, Error)]
    #[error("Withdrawal bundle error")]
    pub struct Error(#[from] Inner);
}
pub use withdrawal_bundle::Error as WithdrawalBundle;
