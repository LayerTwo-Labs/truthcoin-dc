use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, RoTxn, RwTxn};
use std::collections::BTreeMap;

use crate::{state::error::Error, types::Address};

#[derive(Clone)]
pub struct ReputationDbs {
    reputation: DatabaseUnique<SerdeBincode<Address>, SerdeBincode<f64>>,
}

impl ReputationDbs {
    pub const NUM_DBS: u32 = 1;

    pub fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, sneed::env::error::CreateDb> {
        let reputation = DatabaseUnique::create(env, rwtxn, "reputation")?;
        Ok(Self { reputation })
    }

    pub fn get_reputation(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<f64, Error> {
        Ok(self.reputation.try_get(rotxn, address)?.unwrap_or(0.0))
    }

    pub fn get_all_reputations(
        &self,
        rotxn: &RoTxn,
    ) -> Result<BTreeMap<Address, f64>, Error> {
        let mut reputations = BTreeMap::new();
        let mut iter = self.reputation.iter(rotxn)?;
        while let Some((address, reputation)) = iter.next()? {
            if reputation > 0.0 {
                reputations.insert(address, reputation);
            }
        }
        Ok(reputations)
    }

    pub fn set_reputation(
        &self,
        rwtxn: &mut RwTxn,
        address: &Address,
        reputation: f64,
    ) -> Result<(), Error> {
        if reputation <= 0.0 {
            self.reputation.delete(rwtxn, address)?;
        } else {
            self.reputation.put(rwtxn, address, &reputation)?;
        }
        Ok(())
    }

    pub fn delete_reputation(
        &self,
        rwtxn: &mut RwTxn,
        address: &Address,
    ) -> Result<(), Error> {
        self.reputation.delete(rwtxn, address)?;
        Ok(())
    }

    pub fn clear_and_restore(
        &self,
        rwtxn: &mut RwTxn,
        snapshot: &BTreeMap<Address, f64>,
    ) -> Result<(), Error> {
        self.reputation.clear(rwtxn)?;
        for (&address, &rep) in snapshot {
            if rep > 0.0 {
                self.reputation.put(rwtxn, &address, &rep)?;
            }
        }
        Ok(())
    }

    pub fn count_holders(&self, rotxn: &RoTxn) -> Result<u64, Error> {
        let mut iter = self.reputation.iter(rotxn)?;
        let mut count = 0u64;
        while let Some((_, reputation)) = iter.next()? {
            if reputation > 0.0 {
                count += 1;
            }
        }
        Ok(count)
    }
}
