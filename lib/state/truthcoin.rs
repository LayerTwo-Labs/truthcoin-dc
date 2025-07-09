//! Functions and types related to Truthcoin

use std::net::{SocketAddrV4, SocketAddrV6};

use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, db, env};

use crate::{
    state::{
        error::Truthcoin as Error,
        rollback::{RollBack, TxidStamped},
    },
    types::{
        TruthcoinDataUpdates, TruthcoinId, EncryptionPubKey, FilledTransaction,
        Hash, Txid, Update, VerifyingKey,
    },
};

/// Representation of Truthcoin data that supports rollbacks.
/// The most recent datum is the element at the back of the vector.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TruthcoinData {
    /// Commitment to arbitrary data
    pub(in crate::state) commitment: RollBack<TxidStamped<Option<Hash>>>,
    /// Optional ipv4 addr
    pub(in crate::state) socket_addr_v4:
        RollBack<TxidStamped<Option<SocketAddrV4>>>,
    /// Optional ipv6 addr
    pub(in crate::state) socket_addr_v6:
        RollBack<TxidStamped<Option<SocketAddrV6>>>,
    /// Optional pubkey used for encryption
    pub(in crate::state) encryption_pubkey:
        RollBack<TxidStamped<Option<EncryptionPubKey>>>,
    /// Optional pubkey used for signing messages
    pub(in crate::state) signing_pubkey:
        RollBack<TxidStamped<Option<VerifyingKey>>>,
    /// Total supply
    pub(in crate::state) total_supply: RollBack<TxidStamped<u64>>,
}

impl TruthcoinData {
    // initialize from Truthcoin data provided during a registration
    pub(in crate::state) fn init(
        truthcoin_data: crate::types::TruthcoinData,
        initial_supply: u64,
        txid: Txid,
        height: u32,
    ) -> Self {
        Self {
            commitment: RollBack::<TxidStamped<_>>::new(
                truthcoin_data.commitment,
                txid,
                height,
            ),
            socket_addr_v4: RollBack::<TxidStamped<_>>::new(
                truthcoin_data.socket_addr_v4,
                txid,
                height,
            ),
            socket_addr_v6: RollBack::<TxidStamped<_>>::new(
                truthcoin_data.socket_addr_v6,
                txid,
                height,
            ),
            encryption_pubkey: RollBack::<TxidStamped<_>>::new(
                truthcoin_data.encryption_pubkey,
                txid,
                height,
            ),
            signing_pubkey: RollBack::<TxidStamped<_>>::new(
                truthcoin_data.signing_pubkey,
                txid,
                height,
            ),
            total_supply: RollBack::<TxidStamped<_>>::new(
                initial_supply,
                txid,
                height,
            ),
        }
    }

    // apply truthcoin data updates
    pub(in crate::state) fn apply_updates(
        &mut self,
        updates: TruthcoinDataUpdates,
        txid: Txid,
        height: u32,
    ) {
        let Self {
            commitment,
            socket_addr_v4,
            socket_addr_v6,
            encryption_pubkey,
            signing_pubkey,
            total_supply: _,
        } = self;

        // apply an update to a single data field
        fn apply_field_update<T>(
            data_field: &mut RollBack<TxidStamped<Option<T>>>,
            update: Update<T>,
            txid: Txid,
            height: u32,
        ) {
            match update {
                Update::Delete => data_field.push(None, txid, height),
                Update::Retain => (),
                Update::Set(value) => {
                    data_field.push(Some(value), txid, height)
                }
            }
        }
        apply_field_update(commitment, updates.commitment, txid, height);
        apply_field_update(
            socket_addr_v4,
            updates.socket_addr_v4,
            txid,
            height,
        );
        apply_field_update(
            socket_addr_v6,
            updates.socket_addr_v6,
            txid,
            height,
        );
        apply_field_update(
            encryption_pubkey,
            updates.encryption_pubkey,
            txid,
            height,
        );
        apply_field_update(
            signing_pubkey,
            updates.signing_pubkey,
            txid,
            height,
        );
    }

    // revert Truthcoin data updates
    pub(in crate::state) fn revert_updates(
        &mut self,
        updates: TruthcoinDataUpdates,
        txid: Txid,
        height: u32,
    ) {
        // apply an update to a single data field
        fn revert_field_update<T>(
            data_field: &mut RollBack<TxidStamped<Option<T>>>,
            update: Update<T>,
            txid: Txid,
            height: u32,
        ) where
            T: std::fmt::Debug + Eq,
        {
            match update {
                Update::Delete => {
                    let popped = data_field.pop();
                    assert!(popped.is_some());
                    let popped = popped.unwrap();
                    assert!(popped.data.is_none());
                    assert_eq!(popped.txid, txid);
                    assert_eq!(popped.height, height)
                }
                Update::Retain => (),
                Update::Set(value) => {
                    let popped = data_field.pop();
                    assert!(popped.is_some());
                    let popped = popped.unwrap();
                    assert!(popped.data.is_some());
                    assert_eq!(popped.data.unwrap(), value);
                    assert_eq!(popped.txid, txid);
                    assert_eq!(popped.height, height)
                }
            }
        }

        let Self {
            commitment,
            socket_addr_v4,
            socket_addr_v6,
            encryption_pubkey,
            signing_pubkey,
            total_supply: _,
        } = self;
        revert_field_update(
            signing_pubkey,
            updates.signing_pubkey,
            txid,
            height,
        );
        revert_field_update(
            encryption_pubkey,
            updates.encryption_pubkey,
            txid,
            height,
        );
        revert_field_update(
            socket_addr_v6,
            updates.socket_addr_v6,
            txid,
            height,
        );
        revert_field_update(
            socket_addr_v4,
            updates.socket_addr_v4,
            txid,
            height,
        );
        revert_field_update(commitment, updates.commitment, txid, height);
    }

    /** Returns the Truthcoin data as it was, at the specified block height.
     *  If a value was updated several times in the block, returns the
     *  last value seen in the block.
     *  Returns `None` if the data did not exist at the specified block
     *  height. */
    pub fn at_block_height(
        &self,
        height: u32,
    ) -> Option<crate::types::TruthcoinData> {
        Some(crate::types::TruthcoinData {
            commitment: self.commitment.at_block_height(height)?.data,
            socket_addr_v4: self.socket_addr_v4.at_block_height(height)?.data,
            socket_addr_v6: self.socket_addr_v6.at_block_height(height)?.data,
            encryption_pubkey: self
                .encryption_pubkey
                .at_block_height(height)?
                .data,
            signing_pubkey: self.signing_pubkey.at_block_height(height)?.data,
        })
    }

    /// get the current truthcoin data
    pub fn current(&self) -> crate::types::TruthcoinData {
        crate::types::TruthcoinData {
            commitment: self.commitment.latest().data,
            socket_addr_v4: self.socket_addr_v4.latest().data,
            socket_addr_v6: self.socket_addr_v6.latest().data,
            encryption_pubkey: self.encryption_pubkey.latest().data,
            signing_pubkey: self.signing_pubkey.latest().data,
        }
    }
}

/// Truthcoin sequence ID
#[derive(
    utoipa::ToSchema,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SeqId(pub u32);

/// Truthcoin databases
#[derive(Clone)]
pub struct Dbs {
    /// Associates Truthcoin IDs (name hashes) with Truthcoin sequence numbers
    truthcoin_to_seq:
        DatabaseUnique<SerdeBincode<TruthcoinId>, SerdeBincode<SeqId>>,
    /// Associates Truthcoin IDs (name hashes) with Truthcoin data
    // TODO: make this read-only
    truthcoin:
        DatabaseUnique<SerdeBincode<TruthcoinId>, SerdeBincode<TruthcoinData>>,
    /// Associates tx hashes with Truthcoin reservation commitments
    reservations: DatabaseUnique<SerdeBincode<Txid>, SerdeBincode<Hash>>,
    /// Associates Truthcoin sequence numbers with Truthcoin IDs (name hashes)
    // TODO: make this read-only
    seq_to_truthcoin:
        DatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<TruthcoinId>>,
}

impl Dbs {
    pub const NUM_DBS: u32 = 4;

    /// Create / Open DBs. Does not commit the RwTxn.
    pub(in crate::state) fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, env::error::CreateDb> {
        let truthcoin_to_seq =
            DatabaseUnique::create(env, rwtxn, "truthcoin_to_truthcoin_seq")?;
        let truthcoin = DatabaseUnique::create(env, rwtxn, "truthcoin")?;
        let reservations =
            DatabaseUnique::create(env, rwtxn, "truthcoin_reservations")?;
        let seq_to_truthcoin =
            DatabaseUnique::create(env, rwtxn, "truthcoin_seq_to_truthcoin")?;
        Ok(Self {
            reservations,
            seq_to_truthcoin,
            truthcoin_to_seq,
            truthcoin,
        })
    }

    pub fn truthcoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<TruthcoinId>, SerdeBincode<TruthcoinData>>
    {
        &self.truthcoin
    }

    pub fn seq_to_truthcoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<TruthcoinId>> {
        &self.seq_to_truthcoin
    }

    /// The sequence number of the last registered Truthcoin.
    /// Returns `None` if no Truthcoin have been registered.
    pub(in crate::state) fn last_seq(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<SeqId>, db::error::Last> {
        match self.seq_to_truthcoin.last(rotxn)? {
            Some((seq, _)) => Ok(Some(seq)),
            None => Ok(None),
        }
    }

    /// The sequence number that the next registered Truthcoin will take.
    pub(in crate::state) fn next_seq(
        &self,
        rotxn: &RoTxn,
    ) -> Result<SeqId, db::error::Last> {
        match self.last_seq(rotxn)? {
            Some(SeqId(seq)) => Ok(SeqId(seq + 1)),
            None => Ok(SeqId(0)),
        }
    }

    /// Return the Truthcoin data, if it exists
    pub fn try_get_truthcoin(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
    ) -> Result<Option<TruthcoinData>, db::error::TryGet> {
        self.truthcoin.try_get(rotxn, truthcoin)
    }

    /// Return the Truthcoin data. Returns an error if it does not exist.
    pub fn get_truthcoin(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
    ) -> Result<TruthcoinData, Error> {
        self.try_get_truthcoin(rotxn, truthcoin)?
            .ok_or(Error::Missing {
                truthcoin: *truthcoin,
            })
    }

    /// Resolve truthcoin data at the specified block height, if it exists.
    pub fn try_get_truthcoin_data_at_block_height(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
        height: u32,
    ) -> Result<Option<crate::types::TruthcoinData>, db::error::TryGet> {
        let res = self
            .truthcoin
            .try_get(rotxn, truthcoin)?
            .and_then(|truthcoin_data| truthcoin_data.at_block_height(height));
        Ok(res)
    }

    /** Resolve truthcoin data at the specified block height.
     * Returns an error if it does not exist. */
    pub fn get_truthcoin_data_at_block_height(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
        height: u32,
    ) -> Result<crate::types::TruthcoinData, Error> {
        self.get_truthcoin(rotxn, truthcoin)?
            .at_block_height(height)
            .ok_or(Error::MissingData {
                name_hash: truthcoin.0,
                block_height: height,
            })
    }

    /// resolve current truthcoin data, if it exists
    pub fn try_get_current_truthcoin_data(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
    ) -> Result<Option<crate::types::TruthcoinData>, Error> {
        let res = self
            .truthcoin
            .try_get(rotxn, truthcoin)?
            .map(|truthcoin_data| truthcoin_data.current());
        Ok(res)
    }

    /// Resolve current truthcoin data. Returns an error if it does not exist.
    pub fn get_current_truthcoin_data(
        &self,
        rotxn: &RoTxn,
        truthcoin: &TruthcoinId,
    ) -> Result<crate::types::TruthcoinData, Error> {
        self.try_get_current_truthcoin_data(rotxn, truthcoin)?.ok_or(
            Error::Missing {
                truthcoin: *truthcoin,
            },
        )
    }

    /// Delete a Truthcoin reservation.
    /// Returns `true` if a Truthcoin reservation was deleted.
    pub(in crate::state) fn delete_reservation(
        &self,
        rwtxn: &mut RwTxn,
        txid: &Txid,
    ) -> Result<bool, db::error::Delete> {
        self.reservations.delete(rwtxn, txid)
    }

    /// Store a Truthcoin reservation
    pub(in crate::state) fn put_reservation(
        &self,
        rwtxn: &mut RwTxn,
        txid: &Txid,
        commitment: &Hash,
    ) -> Result<(), db::error::Put> {
        self.reservations.put(rwtxn, txid, commitment)
    }

    /// Apply Truthcoin updates
    pub(in crate::state) fn apply_updates(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        truthcoin_updates: TruthcoinDataUpdates,
        height: u32,
    ) -> Result<(), Error> {
        /* The updated Truthcoin is the Truthcoin that corresponds to the last
         * truthcoin output, or equivalently, the Truthcoin corresponding to the
         * last Truthcoin input */
        let updated_truthcoin = filled_tx
            .spent_truthcoin()
            .next_back()
            .ok_or(Error::NoTruthcoinToUpdate)?
            .1
            .truthcoin()
            .expect("should only contain Truthcoin outputs");
        let mut truthcoin_data = self
            .truthcoin
            .try_get(rwtxn, updated_truthcoin)?
            .ok_or(Error::Missing {
                truthcoin: *updated_truthcoin,
            })?;
        truthcoin_data.apply_updates(truthcoin_updates, filled_tx.txid(), height);
        self.truthcoin
            .put(rwtxn, updated_truthcoin, &truthcoin_data)?;
        Ok(())
    }

    /// Revert Truthcoin updates
    pub(in crate::state) fn revert_updates(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        truthcoin_updates: TruthcoinDataUpdates,
        height: u32,
    ) -> Result<(), Error> {
        /* The updated Truthcoin is the Truthcoin that corresponds to the last
         * truthcoin output, or equivalently, the Truthcoin corresponding to the
         * last Truthcoin input */
        let updated_truthcoin = filled_tx
            .spent_truthcoin()
            .next_back()
            .ok_or(Error::NoTruthcoinToUpdate)?
            .1
            .truthcoin()
            .expect("should only contain Truthcoin outputs");
        let mut truthcoin_data = self
            .truthcoin
            .try_get(rwtxn, updated_truthcoin)?
            .ok_or(Error::Missing {
                truthcoin: *updated_truthcoin,
            })?;
        truthcoin_data.revert_updates(
            truthcoin_updates,
            filled_tx.txid(),
            height,
        );
        self.truthcoin
            .put(rwtxn, updated_truthcoin, &truthcoin_data)?;
        Ok(())
    }

    /// Apply Truthcoin registration
    pub(in crate::state) fn apply_registration(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        name_hash: Hash,
        truthcoin_data: &crate::types::TruthcoinData,
        initial_supply: u64,
        height: u32,
    ) -> Result<(), Error> {
        // Find the reservation to burn
        let implied_commitment =
            filled_tx.implied_reservation_commitment().expect(
                "A Truthcoin registration tx should have an implied commitment",
            );
        let burned_reservation_txid =
            filled_tx.spent_reservations().find_map(|(_, filled_output)| {
                let (txid, commitment) = filled_output.reservation_data()
                    .expect("A spent reservation should correspond to a commitment");
                if *commitment == implied_commitment {
                    Some(txid)
                } else {
                    None
                }
            }).expect("A Truthcoin registration tx should correspond to a burned reservation");
        if !self.reservations.delete(rwtxn, burned_reservation_txid)? {
            return Err(Error::MissingReservation {
                txid: *burned_reservation_txid,
            });
        }
        let truthcoin_id = TruthcoinId(name_hash);
        // Assign a sequence number
        {
            let seq = self.next_seq(rwtxn)?;
            self.seq_to_truthcoin.put(rwtxn, &seq, &truthcoin_id)?;
            self.truthcoin_to_seq.put(rwtxn, &truthcoin_id, &seq)?;
        }
        let truthcoin_data = TruthcoinData::init(
            truthcoin_data.clone(),
            initial_supply,
            filled_tx.txid(),
            height,
        );
        self.truthcoin.put(rwtxn, &truthcoin_id, &truthcoin_data)?;
        Ok(())
    }

    /// Revert Truthcoin registration
    pub(in crate::state) fn revert_registration(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        truthcoin: TruthcoinId,
    ) -> Result<(), Error> {
        let Some(seq) = self.truthcoin_to_seq.try_get(rwtxn, &truthcoin)? else {
            return Err(Error::Missing { truthcoin });
        };
        self.truthcoin_to_seq.delete(rwtxn, &truthcoin)?;
        if !self.seq_to_truthcoin.delete(rwtxn, &seq)? {
            return Err(Error::Missing { truthcoin });
        }
        if !self.truthcoin.delete(rwtxn, &truthcoin)? {
            return Err(Error::Missing { truthcoin });
        }
        // Find the reservation to restore
        let implied_commitment =
            filled_tx.implied_reservation_commitment().expect(
                "A Truthcoin registration tx should have an implied commitment",
            );
        let burned_reservation_txid =
            filled_tx.spent_reservations().find_map(|(_, filled_output)| {
                let (txid, commitment) = filled_output.reservation_data()
                    .expect("A spent reservation should correspond to a commitment");
                if *commitment == implied_commitment {
                    Some(txid)
                } else {
                    None
                }
            }).expect("A Truthcoin registration tx should correspond to a burned reservation");
        self.reservations.put(
            rwtxn,
            burned_reservation_txid,
            &implied_commitment,
        )?;
        Ok(())
    }

    /// Apply Truthcoin mint
    pub(in crate::state) fn apply_mint(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        mint_amount: u64,
        height: u32,
    ) -> Result<(), Error> {
        /* The updated Truthcoin is the Truthcoin that corresponds to the last
         * Truthcoin control coin output, or equivalently, the Truthcoin corresponding to the
         * last Truthcoin control coin input */
        let minted_truthcoin = filled_tx
            .spent_truthcoin_controls()
            .next_back()
            .ok_or(Error::NoTruthcoinToMint)?
            .1
            .get_truthcoin()
            .expect("should only contain Truthcoin outputs");
        let mut truthcoin_data = self
            .truthcoin
            .try_get(rwtxn, &minted_truthcoin)?
            .ok_or(Error::Missing {
                truthcoin: minted_truthcoin,
            })?;
        let new_total_supply = truthcoin_data
            .total_supply
            .0
            .first()
            .data
            .checked_add(mint_amount)
            .ok_or(Error::TotalSupplyOverflow)?;
        truthcoin_data.total_supply.push(
            new_total_supply,
            filled_tx.txid(),
            height,
        );
        self.truthcoin
            .put(rwtxn, &minted_truthcoin, &truthcoin_data)?;
        Ok(())
    }

    /// Revert Truthcoin mint
    pub(in crate::state) fn revert_mint(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        mint_amount: u64,
    ) -> Result<(), Error> {
        /* The updated Truthcoin is the Truthcoin that corresponds to the last
         * Truthcoin control coin output, or equivalently, the Truthcoin corresponding to the
         * last Truthcoin control coin input */
        let minted_truthcoin = filled_tx
            .spent_truthcoin_controls()
            .next_back()
            .ok_or(Error::NoTruthcoinToMint)?
            .1
            .get_truthcoin()
            .expect("should only contain Truthcoin outputs");
        let mut truthcoin_data = self
            .truthcoin
            .try_get(rwtxn, &minted_truthcoin)?
            .ok_or(Error::Missing {
                truthcoin: minted_truthcoin,
            })?;
        let total_supply = truthcoin_data.total_supply.0.first().data;
        let _ = truthcoin_data.total_supply.pop();
        let new_total_supply = truthcoin_data.total_supply.0.first().data;
        assert_eq!(
            new_total_supply,
            total_supply
                .checked_sub(mint_amount)
                .ok_or(Error::TotalSupplyUnderflow)?
        );
        self.truthcoin
            .put(rwtxn, &minted_truthcoin, &truthcoin_data)?;
        Ok(())
    }
}
