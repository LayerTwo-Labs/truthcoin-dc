use crate::state::Error;
use crate::state::decisions::DecisionId;
use crate::types::FilledTransaction;
use sneed::RoTxn;

use super::{DecisionValidationInterface, MarketValidator};

pub struct DecisionValidator;

impl DecisionValidator {
    pub fn parse_decision_id_from_hex(
        decision_id_hex: &str,
    ) -> Result<DecisionId, Error> {
        DecisionId::from_hex(decision_id_hex)
    }

    pub fn validate_complete_decision_claim<T>(
        decisions_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: DecisionValidationInterface,
    {
        use crate::state::decisions::DecisionType;

        let claim =
            tx.claim_decision()
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Not a decision claim transaction".to_string(),
                })?;

        match &claim.decision_type {
            DecisionType::Binary | DecisionType::Scaled { .. } => {
                if claim.decisions.len() != 1 {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "{:?} claim must have exactly 1 \
                             decision, got {}",
                            claim.decision_type,
                            claim.decisions.len()
                        ),
                    });
                }
                if claim.decisions[0].option_labels.is_some() {
                    return Err(Error::InvalidTransaction {
                        reason: "option_labels must not be set \
                             for binary/scaled decisions"
                            .to_string(),
                    });
                }
            }
            DecisionType::Category { options } => {
                if claim.decisions.len() != 1 {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Category claim must have exactly \
                             1 decision entry, got {}",
                            claim.decisions.len()
                        ),
                    });
                }
                if options.len() < 2 {
                    return Err(Error::InvalidTransaction {
                        reason: "Category claim requires at \
                             least 2 options"
                            .to_string(),
                    });
                }
                let entry = &claim.decisions[0];
                let entry_labels = entry.option_labels.as_ref().ok_or(
                    Error::InvalidTransaction {
                        reason: "Category claim entry must \
                             have option_labels"
                            .to_string(),
                    },
                )?;
                if entry_labels.len() < 2 {
                    return Err(Error::InvalidTransaction {
                        reason: "Category claim entry must \
                             have at least 2 option_labels"
                            .to_string(),
                    });
                }
                if entry_labels != options {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "option_labels ({}) must match \
                             decision_type options ({})",
                            entry_labels.len(),
                            options.len()
                        ),
                    });
                }
            }
        }

        let market_maker_address = MarketValidator::validate_maker_auth(tx)?;

        let current_ts = decisions_db
            .try_get_mainchain_timestamp(rotxn)?
            .unwrap_or(0);

        let current_height = override_height
            .or_else(|| decisions_db.try_get_height(rotxn).ok().flatten());

        let genesis_ts =
            decisions_db.try_get_genesis_timestamp(rotxn)?.unwrap_or(0);

        for entry in &claim.decisions {
            let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;

            let entry_type = claim.decision_type.clone();

            let decision = crate::state::decisions::Decision::new(
                market_maker_address.0,
                entry_type,
                entry.header.clone(),
                entry.description.clone(),
                entry.option_0_label.clone(),
                entry.option_1_label.clone(),
                entry.tags.clone().unwrap_or_default(),
            )?;

            decisions_db
                .validate_decision_claim(
                    rotxn,
                    decision_id,
                    &decision,
                    current_ts,
                    current_height,
                    genesis_ts,
                )
                .map_err(|e| match e {
                    Error::DecisionNotAvailable {
                        decision_id: _,
                        reason,
                    } => Error::InvalidDecisionId { reason },
                    Error::DecisionAlreadyClaimed { decision_id: _ } => {
                        Error::InvalidDecisionId {
                            reason: format!(
                                "Decision {} already claimed",
                                hex::encode(entry.decision_id_bytes)
                            ),
                        }
                    }
                    other => other,
                })?;
        }

        let mut total_listing_fee: u64 = 0;
        for entry in &claim.decisions {
            let id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            if !id.is_standard() {
                continue;
            }
            let fee = decisions_db.fee_for_decision_id(rotxn, id)?;
            total_listing_fee =
                total_listing_fee.checked_add(fee).ok_or_else(|| {
                    Error::InvalidTransaction {
                        reason: "Listing fee overflow".to_string(),
                    }
                })?;
        }

        if total_listing_fee > 0 {
            let tx_fee = tx
                .bitcoin_fee()
                .map_err(|_| Error::InvalidTransaction {
                    reason: "Failed to compute tx fee".to_string(),
                })?
                .ok_or(Error::NotEnoughValueIn)?;
            if tx_fee.to_sat() < total_listing_fee {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient listing fee: tx fee is \
                         {} sats but listing fee requires \
                         {} sats",
                        tx_fee.to_sat(),
                        total_listing_fee
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_hex_decision_id() {
        let hex = "010001";
        let result = DecisionValidator::parse_decision_id_from_hex(hex);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_invalid_hex_decision_id() {
        assert!(
            DecisionValidator::parse_decision_id_from_hex("zzzzzz").is_err()
        );
    }

    #[test]
    fn parse_wrong_length_decision_id() {
        assert!(DecisionValidator::parse_decision_id_from_hex("01").is_err());
        assert!(
            DecisionValidator::parse_decision_id_from_hex("0100010001")
                .is_err()
        );
    }
}
