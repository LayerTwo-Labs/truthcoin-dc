pub mod database;
pub mod market;
pub mod payouts;
pub mod types;

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests;

pub use crate::math::lmsr::MAX_OUTCOMES;
pub use database::MarketsDatabase;
pub use market::{
    Market, MarketBuilder, compute_market_id, generate_state_combos,
};
pub use payouts::{
    generate_market_author_fee_address, generate_market_treasury_address,
};
pub use types::{
    DEFAULT_MARKET_BETA, DEFAULT_TRADING_FEE, DimensionSpec, FeePayoutRecord,
    FeeRole, MarketError, MarketId, MarketPayoutSummary, MarketState,
    ShareAccount, SharePayoutRecord, parse_dimensions,
};
