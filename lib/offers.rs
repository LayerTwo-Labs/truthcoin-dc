use fallible_iterator::FallibleIterator as _;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, DbError, RoTxn, RwTxn, db};

use crate::{
    state::{self, State},
    types::{Address, OfferId, SignedOffer},
    validation::OfferValidator,
};

pub const MAX_OFFERS_PER_SELLER: usize = 16;

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, thiserror::Error, transitive::Transitive)]
#[transitive(from(db::error::Delete, DbError))]
#[transitive(from(db::error::IterInit, DbError))]
#[transitive(from(db::error::IterItem, DbError))]
#[transitive(from(db::error::Put, DbError))]
#[transitive(from(db::error::TryGet, DbError))]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("invalid offer")]
    Invalid(#[source] Box<state::Error>),
    #[error(
        "seller {seller} already has {MAX_OFFERS_PER_SELLER} offers on the \
         board"
    )]
    SellerLimit { seller: Address },
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::Invalid(Box::new(err))
    }
}

#[derive(Clone)]
pub struct OfferBoard {
    offers: DatabaseUnique<SerdeBincode<OfferId>, SerdeBincode<SignedOffer>>,
}

impl OfferBoard {
    pub const NUM_DBS: u32 = 1;

    pub fn new<Tls>(
        env: &sneed::Env<Tls>,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, sneed::env::error::CreateDb> {
        let offers = DatabaseUnique::create(env, rwtxn, "offer_board")?;
        Ok(Self { offers })
    }

    pub fn get(
        &self,
        rotxn: &RoTxn,
        offer_id: &OfferId,
    ) -> Result<Option<SignedOffer>, Error> {
        Ok(self.offers.try_get(rotxn, offer_id)?)
    }

    pub fn all(&self, rotxn: &RoTxn) -> Result<Vec<SignedOffer>, Error> {
        Ok(self
            .offers
            .iter(rotxn)?
            .map(|(_, offer)| Ok(offer))
            .collect()?)
    }

    pub fn contains(
        &self,
        rotxn: &RoTxn,
        offer_id: &OfferId,
    ) -> Result<bool, Error> {
        Ok(self.offers.try_get(rotxn, offer_id)?.is_some())
    }

    pub fn admit(
        &self,
        rwtxn: &mut RwTxn,
        state: &State,
        signed_offer: &SignedOffer,
    ) -> Result<bool, Error> {
        let offer_id = signed_offer.id();
        if self.contains(rwtxn, &offer_id)? {
            return Ok(false);
        }
        OfferValidator::validate_open_offer(state, rwtxn, signed_offer)?;
        let seller = signed_offer.offer.seller;
        let seller_offers = self
            .all(rwtxn)?
            .iter()
            .filter(|offer| offer.offer.seller == seller)
            .count();
        if seller_offers >= MAX_OFFERS_PER_SELLER {
            return Err(Error::SellerLimit { seller });
        }
        self.offers.put(rwtxn, &offer_id, signed_offer)?;
        Ok(true)
    }

    pub fn prune(&self, rwtxn: &mut RwTxn, state: &State) -> Result<(), Error> {
        let mut stale = Vec::new();
        for signed_offer in self.all(rwtxn)? {
            match OfferValidator::validate_open_offer(
                state,
                rwtxn,
                &signed_offer,
            ) {
                Ok(()) => {}
                Err(state::Error::InvalidTransaction { .. }) => {
                    stale.push(signed_offer.id());
                }
                Err(err) => return Err(err.into()),
            }
        }
        for offer_id in stale {
            self.offers.delete(rwtxn, &offer_id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        state::State,
        types::{
            Votecoin, VotecoinOffer,
            authorization::{SigningKey, get_address},
        },
        validation::offer::sign_test_offer,
    };

    use super::{Error, MAX_OFFERS_PER_SELLER, OfferBoard};

    const GENESIS_TS: u64 = 1_700_000_000;

    /// Dedupes, caps offers per seller, and prunes expired offers
    #[test]
    fn board_admits_and_prunes() {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + OfferBoard::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
        let key = SigningKey::new(&mut rand::rng());
        let seller = get_address(&(&key).into());
        let mut rwtxn = env.write_txn().unwrap();
        let board = OfferBoard::new(&env, &mut rwtxn).unwrap();
        state
            .put_timestamps_for_test(&mut rwtxn, GENESIS_TS, GENESIS_TS)
            .unwrap();
        state
            .reputation()
            .set_reputation(&mut rwtxn, &seller, Votecoin(500_000))
            .unwrap();
        let offer = |nonce| {
            sign_test_offer(
                &key,
                VotecoinOffer {
                    seller,
                    amount: Votecoin(10_000),
                    price: bitcoin::Amount::from_sat(1000),
                    pay_to: seller,
                    voting_period: 1,
                    nonce,
                },
            )
        };

        assert!(board.admit(&mut rwtxn, &state, &offer(0)).unwrap());
        assert!(!board.admit(&mut rwtxn, &state, &offer(0)).unwrap());
        for nonce in 1..MAX_OFFERS_PER_SELLER as u64 {
            assert!(board.admit(&mut rwtxn, &state, &offer(nonce)).unwrap());
        }
        assert!(matches!(
            board.admit(&mut rwtxn, &state, &offer(99)),
            Err(Error::SellerLimit { .. })
        ));

        board.prune(&mut rwtxn, &state).unwrap();
        assert_eq!(board.all(&rwtxn).unwrap().len(), MAX_OFFERS_PER_SELLER);
        state
            .put_timestamps_for_test(
                &mut rwtxn,
                GENESIS_TS,
                GENESIS_TS + crate::types::SECONDS_PER_QUARTER,
            )
            .unwrap();
        board.prune(&mut rwtxn, &state).unwrap();
        assert!(board.all(&rwtxn).unwrap().is_empty());
    }
}
