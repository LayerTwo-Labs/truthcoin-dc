use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, RoTxn, RwTxn};
use std::collections::BTreeMap;

use crate::{
    math::voting::constants::INITIAL_REPUTATION_TOTAL,
    state::{error::Error, voting::from_consensus_scale},
    types::{Address, Body, Votecoin},
};

pub const GENESIS_VOTERS: &[&str] = &[
    "2zH5VDaTZ1gAddyLbgfQQf7yJZmA",
    "3rK8H7KmDwn64dhmG8wWvNj6dYfY",
    "3TUkKbf78JERW2AKd47Z8EUu7cNg",
    "4CMvX6tFgnKYaoHPvf3DGxrXjZiU",
    "3yo5S6okFdMT2c7ULbiiBJzFZP6V",
    "3TUJM5Tm8Ei9oG8De7AVTEMG1Rrn",
    "2fsGbB3adL1y6ha5kCn253Txaw9R",
    "3tW8StaAD2Ji552u4NHB8aVHieVQ",
    "25Gp6Ha7M4LFBc949s5i872oMyw2",
    "ZKANLa7XoAxp75V5GCtiwDNpHgN",
    "41XGiay7vDUM6z6TgsarWvWacBs4",
    "3WSQ8Zmya8YSC5oDQNKqwc7TJp9p",
    "3iSTPQ9xFX3LMVKPAGgHvdVCXzEJ",
    "eHQW6rftVoWFJRgYqpjB5mjyag8",
    "aRqkWcVJcyYhEBgkpW4PjEVyrYr",
    "3tey5gW5VwfUTayXMMof8uGfnLRk",
    "d5H6f6PZXsS8HdefQvQ54bmVM13",
    "ywAxL8Kd4A2YH74geNMQ8xwFEqY",
    "4SstFgR6RgPkhjwWrKP9qtjbnFaB",
    "334mLtrxRuyctRqBufRDSZqayvoQ",
    "3d9DTkAzGXAucR3RaMwM2JfEgFsC",
    "3kMUv6eYTB5pkqjPTk7n8QJJhAr1",
    "2FJUeAaGsszbdQdoEtSU4XrJQyqd",
    "3wt4UWZcDYGLxhcnfNxK7q4i1PFw",
    "2KDnbhi4sfwhUBjzfZkQ8kQ2uaTZ",
    "5BxVACDGZtco8GWFs2jGyVpL6yC",
    "FkFsH74URCScG6sSysLdMZny268",
    "FnUxj5CT4Fb7xF3GdGz6EXsrh7m",
    "3FsXUpsy9DU4nh7GQPmEyKEege7H",
    "3mr5BHuPCSoX2AfPPrEg4dbJwabR",
    "4TwaJ2aoAjWBpYZAiecvqMQnSAm7",
    "3oLjDENbaejvNSEcPw44toCS3wYA",
    "34QNeCoekuRDkJBjSzZjSFKpnse4",
    "3UiDihqBAqMHduLybe3Ei68XDa7Y",
    "2XN9SNHG9dD7BsQarsSoXoGWDHXZ",
    "3XH5kMyzDq1mhRsX7JqpbMxKyvdM",
    "2NYrZFVWwknGhwQZkhsE3sUT2nxg",
    "GuKRLEhjNAGrAVrQGD5aQgXCWfG",
    "nm6Q8MVgcVDuyXDGWtRcfDCpmJS",
    "47EvYE17hSmL7UmKLSyE21r13kJ2",
    "9ZhREEYhALAByVVqCUzEsafAMFg",
    "3hxUnqWUodPS5UwKfZ8LuXRANoUY",
    "4T9RhBT4RihSGp4NeuFqV91VTPVf",
    "2uG7SXnZnQUQHNrR5Z7S8AFVnRtY",
    "2jVgz5m1QdUqJDEsoDMKA9KNMNtx",
    "9twAPmK3m8FUY7Lu8n1w5bD8wdr",
    "4EWWqNQc3Etust3S6jU9b83m3yxS",
    "ZqFiB2hKDjtguK4myL6VoQuEbkn",
    "3Tot36bdX5Hm7EAuHtySyAkH3SLD",
    "2gXSyzL2WefMQ2SEPwv672MgvCoC",
];

fn split_equally(
    voters: &[&str],
) -> Result<BTreeMap<Address, Votecoin>, Error> {
    if voters.is_empty() {
        return Err(Error::EmptyGenesisVoters);
    }
    let share = INITIAL_REPUTATION_TOTAL / voters.len() as f64;
    let shares = voters
        .iter()
        .map(|&address| {
            address
                .parse()
                .map(|parsed| (parsed, share))
                .map_err(|source| Error::InvalidGenesisVoter {
                    address: address.to_owned(),
                    source,
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    from_consensus_scale(shares)
}

pub(in crate::state) fn genesis_allocation(
    genesis_reputation_to_miner: bool,
    body: &Body,
) -> Result<BTreeMap<Address, Votecoin>, Error> {
    if !genesis_reputation_to_miner {
        return split_equally(GENESIS_VOTERS);
    }
    Ok(body
        .coinbase
        .outputs
        .iter()
        .next()
        .map(|output| (output.address, Votecoin::TOTAL))
        .into_iter()
        .collect())
}

#[derive(Clone)]
pub struct ReputationDbs {
    reputation: DatabaseUnique<SerdeBincode<Address>, SerdeBincode<Votecoin>>,
}

impl ReputationDbs {
    pub const NUM_DBS: u32 = 1;

    pub fn new<Tls>(
        env: &sneed::Env<Tls>,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, sneed::env::error::CreateDb> {
        let reputation = DatabaseUnique::create(env, rwtxn, "reputation")?;
        Ok(Self { reputation })
    }

    pub fn get_reputation(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<Votecoin, Error> {
        Ok(self.reputation.try_get(rotxn, address)?.unwrap_or_default())
    }

    pub fn get_all_reputations(
        &self,
        rotxn: &RoTxn,
    ) -> Result<BTreeMap<Address, Votecoin>, Error> {
        let mut reputations = BTreeMap::new();
        let mut iter = self.reputation.iter(rotxn)?;
        while let Some((address, reputation)) = iter.next()? {
            if !reputation.is_zero() {
                reputations.insert(address, reputation);
            }
        }
        Ok(reputations)
    }

    pub fn set_reputation(
        &self,
        rwtxn: &mut RwTxn,
        address: &Address,
        reputation: Votecoin,
    ) -> Result<(), Error> {
        if reputation.is_zero() {
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
        snapshot: &BTreeMap<Address, Votecoin>,
    ) -> Result<(), Error> {
        self.reputation.clear(rwtxn)?;
        for (&address, &rep) in snapshot {
            if !rep.is_zero() {
                self.reputation.put(rwtxn, &address, &rep)?;
            }
        }
        Ok(())
    }

    pub fn count_holders(&self, rotxn: &RoTxn) -> Result<u64, Error> {
        let mut iter = self.reputation.iter(rotxn)?;
        let mut count = 0u64;
        while let Some((_, reputation)) = iter.next()? {
            if !reputation.is_zero() {
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{GENESIS_VOTERS, split_equally};
    use crate::types::{Address, Votecoin};

    /// The genesis whitelist holds fifty distinct valid addresses
    #[test]
    fn genesis_voters_distinct() {
        let addresses = GENESIS_VOTERS
            .iter()
            .map(|address| address.parse::<Address>())
            .collect::<Result<HashSet<_>, _>>()
            .unwrap();
        assert_eq!(GENESIS_VOTERS.len(), 50);
        assert_eq!(addresses.len(), 50);
    }

    /// Splits conserve the total within one unit; bad input errors
    #[test]
    fn genesis_split_equally() {
        let voters: Vec<String> =
            (1..=50).map(|i| Address([i; 20]).as_base58()).collect();
        let voters: Vec<&str> = voters.iter().map(String::as_str).collect();

        let allocation = split_equally(&voters).unwrap();

        assert_eq!(allocation.len(), 50);
        assert!(allocation.values().all(|&share| share == Votecoin(20_000)));
        let seven: Vec<String> =
            (1..=7).map(|i| Address([i; 20]).as_base58()).collect();
        let seven: Vec<&str> = seven.iter().map(String::as_str).collect();
        let uneven = split_equally(&seven).unwrap();
        assert_eq!(
            uneven.values().map(|units| units.0).sum::<u64>(),
            1_000_000
        );
        assert!(uneven.values().all(|units| units.0.abs_diff(142_857) <= 1));
        assert!(split_equally(&[]).is_err());
        assert!(split_equally(&["not an address"]).is_err());
    }
}
