pub mod block;
pub mod decision;
pub mod market;
pub mod vote;

use crate::state::Error;
use crate::state::decisions::{Decision, DecisionId};
use sneed::RoTxn;

pub use block::BlockValidator;
pub use decision::DecisionValidator;
pub use market::{MarketStateValidator, MarketValidator};
pub use vote::{PeriodValidator, VoteValidator};

pub trait DecisionValidationInterface {
    fn validate_decision_claim(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error>;

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;

    fn try_get_genesis_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;

    fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;

    fn get_standard_claimed_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error>;

    fn get_available_decisions(
        &self,
        rotxn: &RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error>;
}
