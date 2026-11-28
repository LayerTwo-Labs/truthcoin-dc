use bitcoin::amount::CheckedSum as _;
use sneed::RoTxn;

use crate::{
    state::{Error, State},
    types::{
        AmountOverflowError, FilledTransaction, GetValue, SignedOffer,
        VotecoinOffer,
    },
    validation::VoteValidator,
};

pub struct OfferValidator;

impl OfferValidator {
    fn invalid(reason: String) -> Error {
        Error::InvalidTransaction { reason }
    }

    fn validate_current_and_unclosed(
        state: &State,
        rotxn: &RoTxn,
        offer: &VotecoinOffer,
    ) -> Result<(), Error> {
        let current_period = state.current_period(rotxn)?;
        if offer.voting_period != current_period {
            return Err(Self::invalid(format!(
                "Offer for period {} has expired (current period {})",
                offer.voting_period, current_period
            )));
        }
        let offer_id = offer.id();
        if state.is_offer_closed(rotxn, &offer_id)? {
            return Err(Self::invalid(format!(
                "Offer {offer_id} was already taken or cancelled"
            )));
        }
        Ok(())
    }

    pub fn validate_open_offer(
        state: &State,
        rotxn: &RoTxn,
        signed_offer: &SignedOffer,
    ) -> Result<(), Error> {
        Self::validate_offer_terms(state, rotxn, signed_offer)?;
        VoteValidator::ensure_reputation_unlocked(
            state,
            rotxn,
            &[signed_offer.offer.seller],
        )
    }

    fn validate_offer_terms(
        state: &State,
        rotxn: &RoTxn,
        signed_offer: &SignedOffer,
    ) -> Result<(), Error> {
        let offer = &signed_offer.offer;
        if !signed_offer.is_signed_by_seller() {
            return Err(Self::invalid(format!(
                "Offer {} is not signed by its seller {}",
                offer.id(),
                offer.seller
            )));
        }
        if offer.amount.is_zero() {
            return Err(Self::invalid("Offer amount must be positive".into()));
        }
        if offer.price == bitcoin::Amount::ZERO {
            return Err(Self::invalid("Offer price must be positive".into()));
        }
        Self::validate_current_and_unclosed(state, rotxn, offer)?;
        let seller_reputation =
            state.reputation().get_reputation(rotxn, &offer.seller)?;
        if seller_reputation < offer.amount {
            return Err(Self::invalid(format!(
                "Seller {} has {seller_reputation} votecoin, offer needs {}",
                offer.seller, offer.amount
            )));
        }
        Ok(())
    }

    pub fn validate_take_offer(
        state: &State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
    ) -> Result<(), Error> {
        let (signed_offer, buyer) =
            filled_tx.take_offer().ok_or_else(|| {
                Self::invalid("Not a take offer transaction".into())
            })?;
        let offer = &signed_offer.offer;
        Self::validate_offer_terms(state, rotxn, signed_offer)?;
        if buyer == offer.seller {
            return Err(Self::invalid("Seller cannot take own offer".into()));
        }
        VoteValidator::ensure_reputation_unlocked(
            state,
            rotxn,
            &[offer.seller, buyer],
        )?;
        let paid = filled_tx
            .transaction
            .outputs
            .iter()
            .filter(|output| {
                output.address == offer.pay_to && output.content.is_value()
            })
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(AmountOverflowError)?;
        if paid < offer.price {
            return Err(Self::invalid(format!(
                "Offer {} pays {paid} to {}, price is {}",
                offer.id(),
                offer.pay_to,
                offer.price
            )));
        }
        let seller_reputation =
            state.reputation().get_reputation(rotxn, &offer.seller)?;
        let buyer_reputation =
            state.reputation().get_reputation(rotxn, &buyer)?;
        VoteValidator::balances_after_transfer(
            seller_reputation,
            buyer_reputation,
            offer.amount,
        )?;
        Ok(())
    }

    pub fn validate_cancel_offer(
        state: &State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
    ) -> Result<(), Error> {
        let offer = filled_tx.cancel_offer().ok_or_else(|| {
            Self::invalid("Not a cancel offer transaction".into())
        })?;
        VoteValidator::validate_actor_identity(filled_tx, offer.seller)?;
        Self::validate_current_and_unclosed(state, rotxn, offer)
    }
}

#[cfg(test)]
pub(crate) fn sign_test_offer(
    key: &crate::types::authorization::SigningKey,
    offer: VotecoinOffer,
) -> SignedOffer {
    use crate::types::{
        Authorization,
        authorization::{self, Dst},
    };

    let msg = offer.signing_message().unwrap();
    SignedOffer {
        offer,
        authorization: Authorization {
            verifying_key: key.into(),
            signature: authorization::sign(
                rand::rng(),
                key,
                Dst::VotecoinOffer,
                &msg,
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        state::State,
        types::{
            Address, FilledTransaction, OutPoint, Output, OutputContent,
            PointedOutputRef, SignedOffer, Transaction, TxData, Txid, Votecoin,
            VotecoinOffer,
            authorization::{SigningKey, get_address},
        },
    };

    use super::{OfferValidator, sign_test_offer};

    const GENESIS_TS: u64 = 1_700_000_000;
    const PRICE_SATS: u64 = 50_000;

    struct Fixture {
        _dir: tempfile::TempDir,
        env: sneed::Env,
        state: State,
        key: SigningKey,
        seller: Address,
        buyer: Address,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024).max_dbs(State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
        let key = SigningKey::new(&mut rand::rng());
        let seller = get_address(&(&key).into());
        let buyer = Address([2; 20]);
        let mut rwtxn = env.write_txn().unwrap();
        state
            .put_timestamps_for_test(&mut rwtxn, GENESIS_TS, GENESIS_TS)
            .unwrap();
        state
            .reputation()
            .set_reputation(&mut rwtxn, &seller, Votecoin(500_000))
            .unwrap();
        state
            .reputation()
            .set_reputation(&mut rwtxn, &buyer, Votecoin(500_000))
            .unwrap();
        rwtxn.commit().unwrap();
        Fixture {
            _dir: dir,
            env,
            state,
            key,
            seller,
            buyer,
        }
    }

    impl Fixture {
        fn offer(&self) -> VotecoinOffer {
            VotecoinOffer {
                seller: self.seller,
                amount: Votecoin(100_000),
                price: bitcoin::Amount::from_sat(PRICE_SATS),
                pay_to: self.seller,
                voting_period: 1,
                nonce: 7,
            }
        }

        fn signed(&self, offer: VotecoinOffer) -> SignedOffer {
            sign_test_offer(&self.key, offer)
        }

        fn validate_take(&self, tx: &FilledTransaction) -> bool {
            let rotxn = self.env.read_txn().unwrap();
            OfferValidator::validate_take_offer(&self.state, &rotxn, tx).is_ok()
        }

        fn validate_cancel(&self, tx: &FilledTransaction) -> bool {
            let rotxn = self.env.read_txn().unwrap();
            OfferValidator::validate_cancel_offer(&self.state, &rotxn, tx)
                .is_ok()
        }
    }

    fn filled(
        payer: Address,
        outputs: Vec<Output>,
        data: TxData,
        actor_address: Option<Address>,
    ) -> FilledTransaction {
        let input = Output {
            address: payer,
            content: OutputContent::Value(bitcoin::Amount::from_sat(100_000)),
        };
        let outpoint = OutPoint::Regular {
            txid: Txid([3; 32]),
            vout: 0,
        };
        let input_hash = crate::types::hash(&PointedOutputRef {
            outpoint,
            output: &input,
        });
        FilledTransaction {
            transaction: Transaction {
                inputs: vec![(outpoint, input_hash)].into(),
                outputs: outputs.into(),
                data: Some(data),
                ..Default::default()
            },
            spent_utxos: vec![input],
            actor_address,
        }
    }

    fn payment(address: Address, sats: u64) -> Output {
        Output {
            address,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    fn take(
        signed: SignedOffer,
        buyer: Address,
        paid_sats: u64,
    ) -> FilledTransaction {
        let pay_to = signed.offer.pay_to;
        filled(
            buyer,
            vec![payment(pay_to, paid_sats)],
            TxData::TakeOffer {
                offer: Box::new(signed),
                buyer,
            },
            None,
        )
    }

    /// A take must pay the price and cannot be made by the seller
    #[test]
    fn take_signed_offer() {
        let f = fixture();
        let signed = f.signed(f.offer());

        assert!(f.validate_take(&take(signed.clone(), f.buyer, PRICE_SATS)));
        assert!(!f.validate_take(&take(
            signed.clone(),
            f.buyer,
            PRICE_SATS - 1
        )));
        assert!(!f.validate_take(&take(signed, f.seller, PRICE_SATS)));
    }

    #[test]
    fn offer_needs_seller_signature() {
        let f = fixture();
        let stranger = SigningKey::new(&mut rand::rng());
        let mut tampered = f.signed(f.offer());
        tampered.offer.amount = Votecoin(200_000);

        assert!(!f.validate_take(&take(
            sign_test_offer(&stranger, f.offer()),
            f.buyer,
            PRICE_SATS
        )));
        assert!(!f.validate_take(&take(tampered, f.buyer, PRICE_SATS)));
    }

    /// Refuses overdraft, zero amount and zero price; accepts the full balance
    #[test]
    fn offer_terms() {
        let f = fixture();
        let offer = f.offer();

        for bad in [
            VotecoinOffer {
                amount: Votecoin(500_001),
                ..offer
            },
            VotecoinOffer {
                amount: Votecoin::ZERO,
                ..offer
            },
            VotecoinOffer {
                price: bitcoin::Amount::ZERO,
                ..offer
            },
        ] {
            let price_sats = bad.price.to_sat();
            assert!(!f.validate_take(&take(
                f.signed(bad),
                f.buyer,
                price_sats
            )));
        }
        let whole_balance = VotecoinOffer {
            amount: Votecoin(500_000),
            ..offer
        };
        assert!(f.validate_take(&take(
            f.signed(whole_balance),
            f.buyer,
            PRICE_SATS
        )));
    }

    #[test]
    fn offer_expires() {
        let f = fixture();
        let signed = f.signed(f.offer());
        let mut rwtxn = f.env.write_txn().unwrap();
        f.state
            .put_timestamps_for_test(
                &mut rwtxn,
                GENESIS_TS,
                GENESIS_TS + crate::types::SECONDS_PER_QUARTER,
            )
            .unwrap();
        rwtxn.commit().unwrap();

        assert!(!f.validate_take(&take(signed, f.buyer, PRICE_SATS)));
    }

    #[test]
    fn only_seller_cancels() {
        let f = fixture();
        let cancel = |payer, actor_address| {
            filled(
                payer,
                vec![],
                TxData::CancelOffer { offer: f.offer() },
                actor_address,
            )
        };

        assert!(f.validate_cancel(&cancel(f.seller, None)));
        assert!(f.validate_cancel(&cancel(f.buyer, Some(f.seller))));
        assert!(!f.validate_cancel(&cancel(f.buyer, None)));
        assert!(!f.validate_cancel(&cancel(f.buyer, Some(f.buyer))));
    }
}
