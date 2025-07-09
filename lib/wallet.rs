use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::Path,
};

use bitcoin::{
    Amount,
    bip32::{ChildNumber, DerivationPath, Xpriv},
};
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::{
    byteorder::BigEndian,
    types::{Bytes, SerdeBincode, Str, U8, U32},
};
use libes::EciesError;
use serde::{Deserialize, Serialize};
use sneed::{DbError, Env, EnvError, RwTxnError, UnitKey, db, env, rwtxn};
use thiserror::Error;
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    authorization::{self, Authorization, Signature, get_address},
    types::{
        Address, AmountOverflowError, AmountUnderflowError, AssetId,
        AuthorizedTransaction, TruthcoinData, TruthcoinId, BitcoinOutputContent,
        DutchAuctionId, DutchAuctionParams, EncryptionPubKey, FilledOutput,
        GetBitcoinValue, Hash, InPoint, OutPoint, Output, OutputContent,
        SpentOutput, Transaction, TxData, VERSION, VerifyingKey, Version,
        WithdrawalOutputContent, keys::Ecies,
    },
    util::Watchable,
};

#[derive(Clone, Debug, Default, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Balance {
    #[serde(rename = "total_sats", with = "bitcoin::amount::serde::as_sat")]
    #[schema(value_type = u64)]
    pub total: Amount,
    #[serde(
        rename = "available_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub available: Amount,
}

#[derive(Debug, Error)]
#[error("Message signature verification key {vk} does not exist")]
pub struct VkDoesNotExistError {
    vk: VerifyingKey,
}

#[allow(clippy::duplicated_attributes)]
#[derive(transitive::Transitive, Debug, Error)]
#[transitive(from(db::error::Delete, DbError))]
#[transitive(from(db::error::IterInit, DbError))]
#[transitive(from(db::error::IterItem, DbError))]
#[transitive(from(db::error::Last, DbError))]
#[transitive(from(db::error::Len, DbError))]
#[transitive(from(db::error::Put, DbError))]
#[transitive(from(db::error::TryGet, DbError))]
#[transitive(from(env::error::CreateDb, EnvError))]
#[transitive(from(env::error::OpenEnv, EnvError))]
#[transitive(from(env::error::ReadTxn, EnvError))]
#[transitive(from(env::error::WriteTxn, EnvError))]
#[transitive(from(rwtxn::error::Commit, RwTxnError))]
pub enum Error {
    #[error("address {address} does not exist")]
    AddressDoesNotExist { address: crate::types::Address },
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("authorization error")]
    Authorization(#[from] crate::authorization::Error),
    #[error("bip32 error")]
    Bip32(#[from] bitcoin::bip32::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("ECIES error: {:?}", .0)]
    Ecies(EciesError),
    #[error("Encryption pubkey {epk} does not exist")]
    EpkDoesNotExist { epk: EncryptionPubKey },
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("no index for address {address}")]
    NoIndex { address: Address },
    #[error(
        "wallet does not have a seed (set with RPC `set-seed-from-mnemonic`)"
    )]
    NoSeed,
    #[error("could not find truthcoin reservation for `{plain_name}`")]
    NoTruthcoinReservation { plain_name: String },
    #[error("not enough funds")]
    NotEnoughFunds,
    #[error("utxo does not exist")]
    NoUtxo,
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[from] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error(transparent)]
    VkDoesNotExist(#[from] Box<VkDoesNotExistError>),
}

/// Marker type for Wallet Env
struct WalletEnv;

type DatabaseUnique<KC, DC> = sneed::DatabaseUnique<KC, DC, WalletEnv>;
type RoTxn<'a> = sneed::RoTxn<'a, WalletEnv>;

#[derive(Clone)]
pub struct Wallet {
    env: sneed::Env<WalletEnv>,
    // Seed is always [u8; 64], but due to serde not implementing serialize
    // for [T; 64], use heed's `Bytes`
    // TODO: Don't store the seed in plaintext.
    seed: DatabaseUnique<U8, Bytes>,
    /// Map each address to it's index
    address_to_index: DatabaseUnique<SerdeBincode<Address>, U32<BigEndian>>,
    /// Map each encryption pubkey to it's index
    epk_to_index:
        DatabaseUnique<SerdeBincode<EncryptionPubKey>, U32<BigEndian>>,
    /// Map each address index to an address
    index_to_address: DatabaseUnique<U32<BigEndian>, SerdeBincode<Address>>,
    /// Map each encryption key index to an encryption pubkey
    index_to_epk:
        DatabaseUnique<U32<BigEndian>, SerdeBincode<EncryptionPubKey>>,
    /// Map each signing key index to a verifying key
    index_to_vk: DatabaseUnique<U32<BigEndian>, SerdeBincode<VerifyingKey>>,
    unconfirmed_utxos:
        DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<FilledOutput>>,
    stxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    spent_unconfirmed_utxos: DatabaseUnique<
        SerdeBincode<OutPoint>,
        SerdeBincode<SpentOutput<OutputContent>>,
    >,
    /// Associates reservation commitments with plaintext Truthcoin names
    truthcoin_reservations: DatabaseUnique<SerdeBincode<[u8; 32]>, Str>,
    /// Associates Truthcoin with plaintext names
    known_truthcoin: DatabaseUnique<SerdeBincode<TruthcoinId>, Str>,
    /// Map each verifying key to it's index
    vk_to_index: DatabaseUnique<SerdeBincode<VerifyingKey>, U32<BigEndian>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 14;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            use heed::EnvFlags;
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS);
            // Apply LMDB "fast" flags consistent with our benchmark setup:
            // - WRITE_MAP lets us write directly into the memory map instead of
            //   copying into LMDB's page buffer, reducing syscall overhead for
            //   write-heavy workloads.
            // - MAP_ASYNC hands dirty-page flushing to the kernel so commits do
            //   not block waiting for msync, keeping latencies tight.
            // - NO_SYNC and NO_META_SYNC skip fsync calls for data and
            //   metadata; this trades durability for throughput, which is
            //   acceptable here because the state can be reconstructed from the
            //   canonical chain if a crash occurs.
            // - NO_READ_AHEAD disables kernel readahead that would otherwise
            //   touch cold pages we immediately overwrite, improving random
            //   access behaviour on SSDs used in testing.
            // - NO_TLS stops LMDB from relying on thread-local storage for
            //   reader slots so transactions can be moved across Tokio tasks.
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
                | EnvFlags::NO_SYNC
                | EnvFlags::NO_META_SYNC
                | EnvFlags::NO_READ_AHEAD
                | EnvFlags::NO_TLS;
            unsafe { env_open_options.flags(fast_flags) };
            unsafe { Env::open(&env_open_options, path) }
                .map_err(EnvError::from)?
        };
        let mut rwtxn = env.write_txn()?;
        let seed_db = DatabaseUnique::create(&env, &mut rwtxn, "seed")?;
        let address_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "address_to_index")?;
        let epk_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "epk_to_index")?;
        let index_to_address =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_address")?;
        let index_to_epk =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_epk")?;
        let index_to_vk =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_vk")?;
        let unconfirmed_utxos =
            DatabaseUnique::create(&env, &mut rwtxn, "unconfirmed_utxos")?;
        let utxos = DatabaseUnique::create(&env, &mut rwtxn, "utxos")?;
        let stxos = DatabaseUnique::create(&env, &mut rwtxn, "stxos")?;
        let spent_unconfirmed_utxos = DatabaseUnique::create(
            &env,
            &mut rwtxn,
            "spent_unconfirmed_utxos",
        )?;
        let truthcoin_reservations =
            DatabaseUnique::create(&env, &mut rwtxn, "truthcoin_reservations")?;
        let known_truthcoin =
            DatabaseUnique::create(&env, &mut rwtxn, "known_truthcoin")?;
        let vk_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "vk_to_index")?;
        let version = DatabaseUnique::create(&env, &mut rwtxn, "version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            env,
            seed: seed_db,
            address_to_index,
            epk_to_index,
            index_to_address,
            index_to_epk,
            index_to_vk,
            unconfirmed_utxos,
            utxos,
            stxos,
            spent_unconfirmed_utxos,
            truthcoin_reservations,
            known_truthcoin,
            vk_to_index,
            _version: version,
        })
    }

    fn get_master_xpriv(&self, rotxn: &RoTxn) -> Result<Xpriv, Error> {
        let seed_bytes = self.seed.try_get(rotxn, &0)?.ok_or(Error::NoSeed)?;
        let res = Xpriv::new_master(bitcoin::NetworkKind::Test, seed_bytes)?;
        Ok(res)
    }

    fn get_encryption_secret(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<x25519_dalek::StaticSecret, Error> {
        let master_xpriv = self.get_master_xpriv(rotxn)?;
        let derivation_path = DerivationPath::master()
            .child(ChildNumber::Hardened { index: 1 })
            .child(ChildNumber::Normal { index });
        let xpriv = master_xpriv
            .derive_priv(&bitcoin::key::Secp256k1::new(), &derivation_path)?;
        let secret = xpriv.private_key.secret_bytes().into();
        Ok(secret)
    }

    /// Get the tx signing key that corresponds to the provided encryption
    /// pubkey
    fn get_encryption_secret_for_epk(
        &self,
        rotxn: &RoTxn,
        epk: &EncryptionPubKey,
    ) -> Result<x25519_dalek::StaticSecret, Error> {
        let epk_idx = self
            .epk_to_index
            .try_get(rotxn, epk)?
            .ok_or(Error::EpkDoesNotExist { epk: *epk })?;
        let encryption_secret = self.get_encryption_secret(rotxn, epk_idx)?;
        // sanity check that encryption secret corresponds to epk
        assert_eq!(*epk, (&encryption_secret).into());
        Ok(encryption_secret)
    }

    fn get_tx_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let master_xpriv = self.get_master_xpriv(rotxn)?;
        let derivation_path = DerivationPath::master()
            .child(ChildNumber::Hardened { index: 0 })
            .child(ChildNumber::Normal { index });
        let xpriv = master_xpriv
            .derive_priv(&bitcoin::key::Secp256k1::new(), &derivation_path)?;
        let signing_key = xpriv.private_key.secret_bytes().into();
        Ok(signing_key)
    }

    /// Get the tx signing key that corresponds to the provided address
    fn get_tx_signing_key_for_addr(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let addr_idx = self
            .address_to_index
            .try_get(rotxn, address)?
            .ok_or(Error::AddressDoesNotExist { address: *address })?;
        let signing_key = self.get_tx_signing_key(rotxn, addr_idx)?;
        // sanity check that signing key corresponds to address
        assert_eq!(*address, get_address(&signing_key.verifying_key().into()));
        Ok(signing_key)
    }

    fn get_message_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let master_xpriv = self.get_master_xpriv(rotxn)?;
        let derivation_path = DerivationPath::master()
            .child(ChildNumber::Hardened { index: 2 })
            .child(ChildNumber::Normal { index });
        let xpriv = master_xpriv
            .derive_priv(&bitcoin::key::Secp256k1::new(), &derivation_path)?;
        let signing_key = xpriv.private_key.secret_bytes().into();
        Ok(signing_key)
    }

    /// Get the tx signing key that corresponds to the provided verifying key
    fn get_message_signing_key_for_vk(
        &self,
        rotxn: &RoTxn,
        vk: &VerifyingKey,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let vk_idx = self
            .vk_to_index
            .try_get(rotxn, vk)?
            .ok_or_else(|| Box::new(VkDoesNotExistError { vk: *vk }))?;
        let signing_key = self.get_message_signing_key(rotxn, vk_idx)?;
        // sanity check that signing key corresponds to vk
        assert_eq!(*vk, signing_key.verifying_key().into());
        Ok(signing_key)
    }

    pub fn get_new_address(&self) -> Result<Address, Error> {
        let mut txn = self.env.write_txn()?;
        let next_index = self
            .index_to_address
            .last(&txn)?
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        let tx_signing_key = self.get_tx_signing_key(&txn, next_index)?;
        let address = get_address(&tx_signing_key.verifying_key().into());
        self.index_to_address.put(&mut txn, &next_index, &address)?;
        self.address_to_index.put(&mut txn, &address, &next_index)?;
        txn.commit()?;
        Ok(address)
    }

    pub fn get_new_encryption_key(&self) -> Result<EncryptionPubKey, Error> {
        let mut txn = self.env.write_txn()?;
        let next_index = self
            .index_to_epk
            .last(&txn)?
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        let encryption_secret = self.get_encryption_secret(&txn, next_index)?;
        let epk = (&encryption_secret).into();
        self.index_to_epk.put(&mut txn, &next_index, &epk)?;
        self.epk_to_index.put(&mut txn, &epk, &next_index)?;
        txn.commit()?;
        Ok(epk)
    }

    /// Get a new message verifying key
    pub fn get_new_verifying_key(&self) -> Result<VerifyingKey, Error> {
        let mut txn = self.env.write_txn()?;
        let next_index = self
            .index_to_vk
            .last(&txn)?
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        let signing_key = self.get_message_signing_key(&txn, next_index)?;
        let vk = signing_key.verifying_key().into();
        self.index_to_vk.put(&mut txn, &next_index, &vk)?;
        self.vk_to_index.put(&mut txn, &vk, &next_index)?;
        txn.commit()?;
        Ok(vk)
    }

    /// Overwrite the seed, or set it if it does not already exist.
    pub fn overwrite_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        self.seed.put(&mut rwtxn, &0, seed).map_err(DbError::from)?;
        self.address_to_index
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.index_to_address
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.utxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.stxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.spent_unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.truthcoin_reservations
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        rwtxn.commit()?;
        Ok(())
    }

    pub fn has_seed(&self) -> Result<bool, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .seed
            .try_get(&rotxn, &0)
            .map_err(DbError::from)?
            .is_some())
    }

    /// Set the seed, if it does not already exist
    pub fn set_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let rotxn = self.env.read_txn()?;
        match self.seed.try_get(&rotxn, &0).map_err(DbError::from)? {
            Some(current_seed) => {
                if current_seed == seed {
                    Ok(())
                } else {
                    Err(Error::SeedAlreadyExists)
                }
            }
            None => {
                drop(rotxn);
                self.overwrite_seed(seed)
            }
        }
    }

    /// Set the seed from a mnemonic seed phrase,
    /// if the seed does not already exist
    pub fn set_seed_from_mnemonic(&self, mnemonic: &str) -> Result<(), Error> {
        let mnemonic =
            bip39::Mnemonic::from_phrase(mnemonic, bip39::Language::English)
                .map_err(Error::ParseMnemonic)?;
        let seed = bip39::Seed::new(&mnemonic, "");
        let seed_bytes: [u8; 64] = seed.as_bytes().try_into().unwrap();
        self.set_seed(&seed_bytes)
    }

    pub fn decrypt_msg(
        &self,
        encryption_pubkey: &EncryptionPubKey,
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let rotxn = self.env.read_txn()?;
        let encryption_secret =
            self.get_encryption_secret_for_epk(&rotxn, encryption_pubkey)?;
        let res = Ecies::decrypt(&encryption_secret, ciphertext)
            .map_err(Error::Ecies)?;
        Ok(res)
    }

    /// Create a transaction with a fee only.
    pub fn create_regular_transaction(
        &self,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let (total, coins) = self.select_bitcoins(fee)?;
        let change = total - fee;
        let inputs = coins.into_keys().collect();
        let outputs = vec![Output::new(
            self.get_new_address()?,
            OutputContent::Bitcoin(BitcoinOutputContent(change)),
        )];
        Ok(Transaction::new(inputs, outputs))
    }

    pub fn create_withdrawal(
        &self,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        value: bitcoin::Amount,
        main_fee: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        tracing::trace!(
            fee = %fee.display_dynamic(),
            ?main_address,
            main_fee = %main_fee.display_dynamic(),
            value = %value.display_dynamic(),
            "Creating withdrawal"
        );
        let (total, coins) = self.select_bitcoins(
            value
                .checked_add(fee)
                .ok_or(AmountOverflowError)?
                .checked_add(main_fee)
                .ok_or(AmountOverflowError)?,
        )?;
        let change = total - value - fee;
        let inputs = coins.into_keys().collect();
        let outputs = vec![
            Output::new(
                self.get_new_address()?,
                OutputContent::Withdrawal(WithdrawalOutputContent {
                    value,
                    main_fee,
                    main_address,
                }),
            ),
            Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ),
        ];
        Ok(Transaction::new(inputs, outputs))
    }

    pub fn create_transfer(
        &self,
        address: Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
        memo: Option<Vec<u8>>,
    ) -> Result<Transaction, Error> {
        let (total, coins) = self.select_bitcoins(
            value.checked_add(fee).ok_or(AmountOverflowError)?,
        )?;
        let change = total - value - fee;
        let inputs = coins.into_keys().collect();
        let mut outputs = vec![Output {
            address,
            content: OutputContent::Bitcoin(BitcoinOutputContent(value)),
            memo: memo.unwrap_or_default(),
        }];
        if change != Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ))
        }
        Ok(Transaction::new(inputs, outputs))
    }

    pub fn create_truthcoin_transfer(
        &self,
        address: Address,
        asset_id: TruthcoinId,
        amount: u64,
        fee: bitcoin::Amount,
        memo: Option<Vec<u8>>,
    ) -> Result<Transaction, Error> {
        let (total_sats, bitcoins) = self.select_bitcoins(fee)?;
        let change_sats = total_sats - fee;
        let mut inputs: Vec<_> = bitcoins.into_keys().collect();
        let (total_truthcoin, truthcoin_utxos) =
            self.select_truthcoin_utxos(asset_id, amount)?;
        let truthcoin_change = total_truthcoin - amount;
        inputs.extend(truthcoin_utxos.into_keys());
        let mut outputs = vec![Output {
            address,
            content: OutputContent::Truthcoin(amount),
            memo: memo.unwrap_or_default(),
        }];
        if change_sats != Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change_sats)),
            ))
        }
        if truthcoin_change != 0 {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Truthcoin(truthcoin_change),
            ))
        }
        Ok(Transaction::new(inputs, outputs))
    }

    /// given a regular transaction, add a truthcoin reservation.
    /// given a truthcoin reservation tx, change the reserved name.
    /// panics if the tx is not regular or a truthcoin reservation tx.
    pub fn reserve_truthcoin(
        &self,
        tx: &mut Transaction,
        plain_name: &str,
    ) -> Result<(), Error> {
        assert!(
            tx.is_regular() || tx.is_reservation(),
            "this function only accepts a regular or truthcoin reservation tx"
        );
        // address for the reservation output
        let reservation_addr =
            // if the tx is already truthcoin reservation,
            // re-use the reservation address
            if tx.is_reservation() {
                tx.reservation_outputs().next_back()
                    .expect("A truthcoin reservation tx must have at least one reservation output")
                    .address
            }
            // if the last output is owned by this wallet, then use
            // the address associated with the last output
            else if let Some(last_output) = tx.outputs.last() {
                let last_output_addr = last_output.address;
                let rotxn = self.env.read_txn()?;
                if self.address_to_index.try_get(&rotxn, &last_output_addr)?.is_some() {
                    last_output_addr
                } else {
                    self.get_new_address()?
                }
            } else {
                self.get_new_address()?
            };
        let rotxn = self.env.read_txn()?;
        let reservation_signing_key =
            self.get_tx_signing_key_for_addr(&rotxn, &reservation_addr)?;
        let name_hash: Hash = blake3::hash(plain_name.as_bytes()).into();
        let truthcoin_id = TruthcoinId(name_hash);
        // hmac(secret, name_hash)
        let nonce =
            blake3::keyed_hash(reservation_signing_key.as_bytes(), &name_hash)
                .into();
        // hmac(nonce, name_hash)
        let commitment = blake3::keyed_hash(&nonce, &name_hash).into();
        // store reservation data
        let mut rwtxn = self.env.write_txn()?;
        self.truthcoin_reservations
            .put(&mut rwtxn, &commitment, plain_name)?;
        self.known_truthcoin
            .put(&mut rwtxn, &truthcoin_id, plain_name)?;
        rwtxn.commit()?;
        // if the tx is regular, add a reservation output
        if tx.is_regular() {
            let reservation_output = Output::new(
                reservation_addr,
                OutputContent::TruthcoinReservation,
            );
            tx.outputs.push(reservation_output);
        };
        tx.data = Some(TxData::TruthcoinReservation { commitment });
        Ok(())
    }

    /// given a regular transaction, add a truthcoin registration.
    /// panics if the tx is not regular.
    /// returns an error if there is no corresponding reservation utxo
    /// does not modify the tx if there is no corresponding reservation utxo.
    pub fn register_truthcoin(
        &self,
        tx: &mut Transaction,
        plain_name: &str,
        truthcoin_data: Cow<TruthcoinData>,
        initial_supply: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        // address for the registration output
        let registration_addr =
            // if the last output is owned by this wallet, then use
            // the address associated with the last output
            if let Some(last_output) = tx.outputs.last() {
                let last_output_addr = last_output.address;
                let rotxn = self.env.read_txn()?;
                if self.address_to_index.try_get(&rotxn, &last_output_addr)?.is_some() {
                    last_output_addr
                } else {
                    self.get_new_address()?
                }
            } else {
                self.get_new_address()?
            };
        let name_hash: Hash = blake3::hash(plain_name.as_bytes()).into();
        let truthcoin_id = TruthcoinId(name_hash);
        /* Search for reservation utxo by the following procedure:
        For each reservation:
        * Get the corresponding signing key
        * Compute a reservation commitment for the truthcoin to be registered
        * If the computed commitment is the same as the reservation commitment,
          then use this utxo. Otherwise, continue */
        // outpoint and nonce, if found
        let mut reservation_outpoint_nonce: Option<(OutPoint, Hash)> = None;
        for (outpoint, filled_output) in self.get_utxos()?.into_iter() {
            if let Some(reservation_commitment) =
                filled_output.reservation_commitment()
            {
                // for each reservation, get the signing key, and
                let reservation_addr = filled_output.address;
                let rotxn = self.env.read_txn()?;
                let reservation_signing_key = self
                    .get_tx_signing_key_for_addr(&rotxn, &reservation_addr)?;
                // hmac(secret, name_hash)
                let nonce = blake3::keyed_hash(
                    reservation_signing_key.as_bytes(),
                    &name_hash,
                )
                .into();
                // hmac(nonce, name_hash)
                let commitment = blake3::keyed_hash(&nonce, &name_hash);
                // WARNING: This comparison MUST be done in constant time.
                // `blake3::Hash` handles this; DO NOT compare as byte arrays
                if commitment == *reservation_commitment {
                    reservation_outpoint_nonce = Some((outpoint, nonce));
                    break;
                }
            }
        }
        // store truthcoin data
        let mut rwtxn = self.env.write_txn()?;
        self.known_truthcoin
            .put(&mut rwtxn, &truthcoin_id, plain_name)?;
        rwtxn.commit()?;
        let (reservation_outpoint, nonce) = reservation_outpoint_nonce
            .ok_or_else(|| Error::NoTruthcoinReservation {
                plain_name: plain_name.to_owned(),
            })?;
        tx.inputs.push(reservation_outpoint);
        if initial_supply != 0 {
            let mint_output = Output::new(
                registration_addr,
                OutputContent::Truthcoin(initial_supply),
            );
            tx.outputs.push(mint_output);
        };
        let control_coin_output =
            Output::new(registration_addr, OutputContent::TruthcoinControl);
        tx.outputs.push(control_coin_output);
        tx.data = Some(TxData::TruthcoinRegistration {
            name_hash,
            revealed_nonce: nonce,
            truthcoin_data: Box::new(truthcoin_data.into_owned()),
            initial_supply,
        });
        Ok(())
    }

    pub fn select_bitcoins(
        &self,
        value: bitcoin::Amount,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;
        let mut bitcoin_utxos: Vec<(_, Output)> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .filter_map(|(outpoint, output)| {
                if output.is_bitcoin() {
                    Ok(Some((outpoint, output.into())))
                } else {
                    Ok(None)
                }
            })
            .collect()
            .map_err(DbError::from)?;
        bitcoin_utxos
            .sort_unstable_by_key(|(_, output)| output.get_bitcoin_value());
        let mut unconfirmed_bitcoin_utxos: Vec<_> = self
            .unconfirmed_utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .filter(|(_outpoint, output)| Ok(output.is_bitcoin()))
            .collect()
            .map_err(DbError::from)?;
        unconfirmed_bitcoin_utxos
            .sort_unstable_by_key(|(_, output)| output.get_bitcoin_value());

        let mut selected = HashMap::new();
        let mut total = bitcoin::Amount::ZERO;
        for (outpoint, output) in
            bitcoin_utxos.into_iter().chain(unconfirmed_bitcoin_utxos)
        {
            if output.content.is_withdrawal()
                || output.is_truthcoin()
                || output.is_reservation()
                || output.get_bitcoin_value() == bitcoin::Amount::ZERO
            {
                continue;
            }
            if total >= value {
                break;
            }
            total = total
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(outpoint, output.clone());
        }
        if total >= value {
            Ok((total, selected))
        } else {
            Err(Error::NotEnoughFunds)
        }
    }

    // Select UTXOs for the specified Truthcoin
    pub fn select_truthcoin_utxos(
        &self,
        truthcoin: TruthcoinId,
        value: u64,
    ) -> Result<(u64, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;
        let mut truthcoin_utxos: Vec<_> = self
            .utxos
            .iter(&rotxn)?
            .filter(|(_outpoint, output)| {
                Ok(output.truthcoin().is_some_and(|output_truthcoin| {
                    truthcoin == *output_truthcoin
                }))
            })
            .collect()?;
        truthcoin_utxos.sort_unstable_by_key(|(_, output)| {
            output
                .truthcoin_value()
                .map(|(_, truthcoin_value)| truthcoin_value)
        });

        let mut selected = HashMap::new();
        let mut total_value: u64 = 0;
        for (outpoint, output) in &truthcoin_utxos {
            if output.content.is_withdrawal() {
                continue;
            }
            if total_value > value {
                break;
            }
            let (_, truthcoin_value) = output.truthcoin_value().unwrap();
            total_value += truthcoin_value;
            selected.insert(*outpoint, output.clone().into());
        }
        if total_value < value {
            return Err(Error::NotEnoughFunds);
        }
        Ok((total_value, selected))
    }

    // Select control coin for the specified Truthcoin
    pub fn select_truthcoin_control(
        &self,
        truthcoin: TruthcoinId,
    ) -> Result<(OutPoint, Output), Error> {
        let rotxn = self.env.read_txn()?;
        let truthcoin_utxo =
            self.utxos.iter(&rotxn)?.find_map(|(outpoint, output)| {
                if let Some(output_truthcoin) = output.truthcoin()
                    && truthcoin == *output_truthcoin
                {
                    Ok(Some((outpoint, output.into())))
                } else {
                    Ok(None)
                }
            })?;
        truthcoin_utxo.ok_or(Error::NotEnoughFunds)
    }

    pub fn select_asset_utxos(
        &self,
        asset: AssetId,
        amount: u64,
    ) -> Result<(u64, HashMap<OutPoint, Output>), Error> {
        match asset {
            AssetId::Bitcoin => self
                .select_bitcoins(bitcoin::Amount::from_sat(amount))
                .map(|(amount, utxos)| (amount.to_sat(), utxos)),
            AssetId::Truthcoin(truthcoin) => {
                self.select_truthcoin_utxos(truthcoin, amount)
            }
            AssetId::TruthcoinControl(truthcoin) => {
                if amount == 1 {
                    let utxo = self.select_truthcoin_control(truthcoin)?;
                    Ok((1, HashMap::from_iter([utxo])))
                } else {
                    do yeet Error::NotEnoughFunds
                }
            }
        }
    }

    // Select control coin for the specified LP token
    pub fn select_amm_lp_tokens(
        &self,
        asset0: AssetId,
        asset1: AssetId,
        amount: u64,
    ) -> Result<(u64, HashMap<OutPoint, FilledOutput>), Error> {
        let rotxn = self.env.read_txn()?;
        let mut amm_lp_token_utxos: Vec<_> = self
            .utxos
            .iter(&rotxn)?
            .filter(|(_outpoint, output)| {
                Ok(output.lp_token_amount().is_some_and(
                    |(pool_asset0, pool_asset1, _)| {
                        pool_asset0 == asset0 && pool_asset1 == asset1
                    },
                ))
            })
            .collect()?;
        amm_lp_token_utxos.sort_unstable_by_key(|(_, output)| {
            output.lp_token_amount().map(|(_, _, amount)| amount)
        });
        let mut selected = HashMap::new();
        let mut total_amount: u64 = 0;
        for (outpoint, output) in &amm_lp_token_utxos {
            if total_amount > amount {
                break;
            }
            let (_, _, lp_token_amount) = output.lp_token_amount().unwrap();
            total_amount += lp_token_amount;
            selected.insert(*outpoint, output.clone());
        }
        if total_amount < amount {
            return Err(Error::NotEnoughFunds);
        }
        Ok((total_amount, selected))
    }

    // Select dutch auction receipt utxo for the specified auction
    pub fn select_dutch_auction_receipt(
        &self,
        auction_id: DutchAuctionId,
    ) -> Result<(OutPoint, FilledOutput), Error> {
        let rotxn = self.env.read_txn()?;
        /*
        let mut receipt_utxo = None;
        for item in self.utxos.iter(&txn)? {
            let (outpoint, output) = item?;
            if let Some(output_auction_id) = output.dutch_auction_receipt()
                && auction_id == output_auction_id
            {
                receipt_utxo = Some((outpoint, output));
                break;
            }
        }
        */
        let receipt_utxo =
            self.utxos.iter(&rotxn)?.find(|(_outpoint, output)| {
                Ok(output.dutch_auction_receipt().is_some_and(
                    |output_auction_id| auction_id == output_auction_id,
                ))
            })?;
        receipt_utxo.ok_or(Error::NotEnoughFunds)
    }

    /// Given a regular transaction, add an AMM mint.
    pub fn amm_mint(
        &self,
        tx: &mut Transaction,
        asset0: AssetId,
        asset1: AssetId,
        amount0: u64,
        amount1: u64,
        lp_token_mint: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        // address for the LP token output
        let lp_token_addr = self.get_new_address()?;

        let (input_amount0, asset0_utxos) =
            self.select_asset_utxos(asset0, amount0)?;
        let (input_amount1, asset1_utxos) =
            self.select_asset_utxos(asset1, amount1)?;

        let change_amount0 = input_amount0 - amount0;
        let change_amount1 = input_amount1 - amount1;
        let change_output0 = if change_amount0 != 0 {
            let address = self.get_new_address()?;
            let content = match asset0 {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(change_amount0),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(change_amount0),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let change_output1 = if change_amount1 != 0 {
            let address = self.get_new_address()?;
            let content = match asset1 {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(change_amount1),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(change_amount1),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let lp_token_output = Output {
            address: lp_token_addr,
            content: OutputContent::AmmLpToken(lp_token_mint),
            memo: Vec::new(),
        };

        /* The first two unique assets in the inputs must be
         * `asset0` and `asset1` */
        tx.inputs.extend(asset0_utxos.keys());
        tx.inputs.extend(asset1_utxos.keys());
        tx.inputs
            .rotate_right(asset0_utxos.len() + asset1_utxos.len());

        tx.outputs.extend(change_output0);
        tx.outputs.extend(change_output1);
        tx.outputs.push(lp_token_output);

        tx.data = Some(TxData::AmmMint {
            amount0,
            amount1,
            lp_token_mint,
        });
        Ok(())
    }

    // Given a regular transaction, add an AMM burn.
    pub fn amm_burn(
        &self,
        tx: &mut Transaction,
        asset0: AssetId,
        asset1: AssetId,
        amount0: u64,
        amount1: u64,
        lp_token_burn: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        // address for receiving asset0
        let asset0_addr = self.get_new_address()?;
        // address for receiving asset1
        let asset1_addr = self.get_new_address()?;

        let (input_lp_token_amount, lp_token_utxos) =
            self.select_amm_lp_tokens(asset0, asset1, lp_token_burn)?;

        let lp_token_change_amount = input_lp_token_amount - lp_token_burn;
        let lp_token_change_output = if lp_token_change_amount != 0 {
            let address = self.get_new_address()?;
            Some(Output {
                address,
                content: OutputContent::AmmLpToken(lp_token_change_amount),
                memo: Vec::new(),
            })
        } else {
            None
        };
        let asset0_output = Output {
            address: asset0_addr,
            memo: Vec::new(),
            content: match asset0 {
                AssetId::Bitcoin => OutputContent::Bitcoin(
                    BitcoinOutputContent(bitcoin::Amount::from_sat(amount0)),
                ),
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount0),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            },
        };
        let asset1_output = Output {
            address: asset1_addr,
            memo: Vec::new(),
            content: match asset1 {
                AssetId::Bitcoin => OutputContent::Bitcoin(
                    BitcoinOutputContent(bitcoin::Amount::from_sat(amount1)),
                ),
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount1),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            },
        };

        /* The AMM lp token input must occur before any other AMM lp token
         * inputs. */
        tx.inputs.extend(lp_token_utxos.keys());
        tx.inputs.rotate_right(lp_token_utxos.len());

        tx.outputs.extend(lp_token_change_output);
        tx.outputs.push(asset0_output);
        tx.outputs.push(asset1_output);

        tx.data = Some(TxData::AmmBurn {
            amount0,
            amount1,
            lp_token_burn,
        });
        Ok(())
    }

    // Given a regular transaction, add an AMM swap.
    pub fn amm_swap(
        &self,
        tx: &mut Transaction,
        asset_spend: AssetId,
        asset_receive: AssetId,
        amount_spend: u64,
        amount_receive: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        // Address for receiving `asset_receive`
        let receive_addr = self.get_new_address()?;
        let (input_amount_spend, spend_utxos) =
            self.select_asset_utxos(asset_spend, amount_spend)?;
        let amount_change = input_amount_spend - amount_spend;
        let change_output = if amount_change != 0 {
            let address = self.get_new_address()?;
            let content = match asset_spend {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(amount_change),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount_change),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let receive_output = Output {
            address: receive_addr,
            memo: Vec::new(),
            content: match asset_receive {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(amount_receive),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount_receive),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            },
        };
        // The first unique asset in the inputs must be `asset_spend`.
        tx.inputs.extend(spend_utxos.keys());
        tx.inputs.rotate_right(spend_utxos.len());
        tx.outputs.extend(change_output);
        tx.outputs.push(receive_output);
        tx.data = Some(TxData::AmmSwap {
            amount_spent: amount_spend,
            amount_receive,
            pair_asset: asset_receive,
        });
        Ok(())
    }

    /// Given a regular transaction, create a dutch auction tx
    pub fn dutch_auction_create(
        &self,
        tx: &mut Transaction,
        dutch_auction_params: DutchAuctionParams,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        let (input_base_amount, base_utxos) = self.select_asset_utxos(
            dutch_auction_params.base_asset,
            dutch_auction_params.base_amount,
        )?;
        let change_amount =
            input_base_amount - dutch_auction_params.base_amount;
        let change_output = if change_amount != 0 {
            let address = self.get_new_address()?;
            let content = match dutch_auction_params.base_asset {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(change_amount),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(change_amount),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let dutch_auction_receipt = Output {
            address: self.get_new_address()?,
            memo: Vec::new(),
            content: OutputContent::DutchAuctionReceipt,
        };
        tx.inputs.extend(base_utxos.keys());
        tx.outputs.extend(change_output);
        tx.outputs.push(dutch_auction_receipt);
        tx.data = Some(TxData::DutchAuctionCreate(dutch_auction_params));
        Ok(())
    }

    /// Given a regular transaction, create a dutch auction bid
    pub fn dutch_auction_bid(
        &self,
        tx: &mut Transaction,
        auction_id: DutchAuctionId,
        base_asset: AssetId,
        quote_asset: AssetId,
        bid_size: u64,
        receive_quantity: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        let (input_quote_amount, quote_utxos) =
            self.select_asset_utxos(quote_asset, bid_size)?;
        let change_amount = input_quote_amount - bid_size;
        let change_output = if change_amount != 0 {
            let address = self.get_new_address()?;
            let content = match quote_asset {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(change_amount),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(change_amount),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let base_output = {
            let content = match base_asset {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(receive_quantity),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(change_amount),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Output {
                address: self.get_new_address()?,
                memo: Vec::new(),
                content,
            }
        };

        // The first unique asset in the inputs must be `quote_asset`.
        tx.inputs.extend(quote_utxos.keys());
        tx.inputs.rotate_right(quote_utxos.len());

        tx.outputs.push(base_output);
        tx.outputs.extend(change_output);
        tx.data = Some(TxData::DutchAuctionBid {
            auction_id,
            receive_asset: base_asset,
            quantity: receive_quantity,
            bid_size,
        });
        Ok(())
    }

    /// Given a regular transaction, create a dutch auction collect tx
    pub fn dutch_auction_collect(
        &self,
        tx: &mut Transaction,
        auction_id: DutchAuctionId,
        base_asset: AssetId,
        quote_asset: AssetId,
        amount_base: u64,
        amount_quote: u64,
    ) -> Result<(), Error> {
        assert!(tx.is_regular(), "this function only accepts a regular tx");
        let (dutch_auction_receipt_input, _) =
            self.select_dutch_auction_receipt(auction_id)?;
        let base_output = if amount_base != 0 {
            let address = self.get_new_address()?;
            let content = match base_asset {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(amount_base),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount_base),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };
        let quote_output = if amount_quote != 0 {
            let address = self.get_new_address()?;
            let content = match quote_asset {
                AssetId::Bitcoin => {
                    OutputContent::Bitcoin(BitcoinOutputContent(
                        bitcoin::Amount::from_sat(amount_quote),
                    ))
                }
                AssetId::Truthcoin(_) => OutputContent::Truthcoin(amount_quote),
                AssetId::TruthcoinControl(_) => OutputContent::TruthcoinControl,
            };
            Some(Output {
                address,
                memo: Vec::new(),
                content,
            })
        } else {
            None
        };

        /* The Dutch auction receipt must occur before any other Dutch auction
        receipts in the inputs. */
        tx.inputs.push(dutch_auction_receipt_input);
        tx.inputs.rotate_right(1);

        tx.outputs.extend(base_output);
        tx.outputs.extend(quote_output);
        tx.data = Some(TxData::DutchAuctionCollect {
            asset_offered: base_asset,
            asset_receive: quote_asset,
            amount_offered_remaining: amount_base,
            amount_received: amount_quote,
        });
        Ok(())
    }

    pub fn spend_utxos(
        &self,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        for (outpoint, inpoint) in spent {
            if let Some(output) = self
                .utxos
                .try_get(&rwtxn, outpoint)
                .map_err(DbError::from)?
            {
                self.utxos
                    .delete(&mut rwtxn, outpoint)
                    .map_err(DbError::from)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos
                    .put(&mut rwtxn, outpoint, &spent_output)
                    .map_err(DbError::from)?;
            } else if let Some(output) =
                self.unconfirmed_utxos.try_get(&rwtxn, outpoint)?
            {
                self.unconfirmed_utxos.delete(&mut rwtxn, outpoint)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.spent_unconfirmed_utxos.put(
                    &mut rwtxn,
                    outpoint,
                    &spent_output,
                )?;
            } else {
                continue;
            }
        }
        rwtxn.commit()?;
        Ok(())
    }

    pub fn put_unconfirmed_utxos(
        &self,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;
        for (outpoint, output) in utxos {
            self.unconfirmed_utxos.put(&mut txn, outpoint, output)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn put_utxos(
        &self,
        utxos: &HashMap<OutPoint, FilledOutput>,
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        for (outpoint, output) in utxos {
            self.utxos
                .put(&mut rwtxn, outpoint, output)
                .map_err(DbError::from)?;
        }
        rwtxn.commit()?;
        Ok(())
    }

    pub fn get_bitcoin_balance(&self) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let rotxn = self.env.read_txn()?;
        let () = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, utxo)| {
                let value = utxo.get_bitcoin_value();
                balance.total = balance
                    .total
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                if !utxo.content.is_withdrawal() {
                    balance.available = balance
                        .available
                        .checked_add(value)
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        Ok(balance)
    }

    /// gets the plaintext name associated with a truthcoin reservation
    /// commitment, if it is known by the wallet.
    pub fn get_truthcoin_reservation_plaintext(
        &self,
        commitment: &Hash,
    ) -> Result<Option<String>, Error> {
        let rotxn = self.env.read_txn()?;
        let res = self.truthcoin_reservations.try_get(&rotxn, commitment)?;
        Ok(res.map(String::from))
    }

    /// gets the plaintext name associated with a truthcoin,
    /// if it is known by the wallet.
    pub fn get_truthcoin_plaintext(
        &self,
        truthcoin: &TruthcoinId,
    ) -> Result<Option<String>, Error> {
        let rotxn = self.env.read_txn()?;
        let res = self.known_truthcoin.try_get(&rotxn, truthcoin)?;
        Ok(res.map(String::from))
    }

    pub fn get_utxos(&self) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let rotxn = self.env.read_txn()?;
        let utxos: HashMap<_, _> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;

        Ok(utxos)
    }

    pub fn get_unconfirmed_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn()?;
        let utxos = self.unconfirmed_utxos.iter(&rotxn)?.collect()?;
        Ok(utxos)
    }

    pub fn get_stxos(&self) -> Result<HashMap<OutPoint, SpentOutput>, Error> {
        let rotxn = self.env.read_txn()?;
        let stxos = self.stxos.iter(&rotxn)?.collect()?;
        Ok(stxos)
    }

    pub fn get_spent_unconfirmed_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, SpentOutput<OutputContent>>, Error> {
        let rotxn = self.env.read_txn()?;
        let stxos = self.spent_unconfirmed_utxos.iter(&rotxn)?.collect()?;
        Ok(stxos)
    }

    /// get all owned truthcoin utxos
    pub fn get_truthcoin(
        &self,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let mut utxos = self.get_utxos()?;
        utxos.retain(|_, output| output.is_truthcoin());
        Ok(utxos)
    }

    /// get all spent truthcoin utxos
    pub fn get_spent_truthcoin(
        &self,
    ) -> Result<HashMap<OutPoint, SpentOutput>, Error> {
        let mut stxos = self.get_stxos()?;
        stxos.retain(|_, output| output.output.is_truthcoin());
        Ok(stxos)
    }

    pub fn get_addresses(&self) -> Result<HashSet<Address>, Error> {
        let rotxn = self.env.read_txn()?;
        let addresses: HashSet<_> = self
            .index_to_address
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(_, address)| Ok(address))
            .collect()
            .map_err(DbError::from)?;
        Ok(addresses)
    }

    pub fn authorize(
        &self,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error> {
        let rotxn = self.env.read_txn()?;
        let mut authorizations = vec![];
        for input in &transaction.inputs {
            let spent_utxo = if let Some(utxo) =
                self.utxos.try_get(&rotxn, input).map_err(DbError::from)?
            {
                utxo.into()
            } else if let Some(utxo) = self
                .unconfirmed_utxos
                .try_get(&rotxn, input)
                .map_err(DbError::from)?
            {
                utxo
            } else {
                return Err(Error::NoUtxo);
            };
            let index = self
                .address_to_index
                .try_get(&rotxn, &spent_utxo.address)
                .map_err(DbError::from)?
                .ok_or(Error::NoIndex {
                    address: spent_utxo.address,
                })?;
            let tx_signing_key = self.get_tx_signing_key(&rotxn, index)?;
            let signature =
                crate::authorization::sign_tx(&tx_signing_key, &transaction)?;
            authorizations.push(Authorization {
                verifying_key: tx_signing_key.verifying_key().into(),
                signature,
            });
        }
        Ok(AuthorizedTransaction {
            authorizations,
            transaction,
        })
    }

    pub fn get_num_addresses(&self) -> Result<u32, Error> {
        let rotxn = self.env.read_txn()?;
        let res = self.index_to_address.len(&rotxn)? as u32;
        Ok(res)
    }

    pub fn sign_arbitrary_msg(
        &self,
        verifying_key: &VerifyingKey,
        msg: &str,
    ) -> Result<Signature, Error> {
        use authorization::{Dst, sign};
        let rotxn = self.env.read_txn()?;
        let signing_key =
            self.get_message_signing_key_for_vk(&rotxn, verifying_key)?;
        let res = sign(&signing_key, Dst::Arbitrary, msg.as_bytes());
        Ok(res)
    }

    pub fn sign_arbitrary_msg_as_addr(
        &self,
        address: &Address,
        msg: &str,
    ) -> Result<Authorization, Error> {
        use authorization::{Dst, sign};
        let rotxn = self.env.read_txn()?;
        let signing_key = self.get_tx_signing_key_for_addr(&rotxn, address)?;
        let signature = sign(&signing_key, Dst::Arbitrary, msg.as_bytes());
        let verifying_key = signing_key.verifying_key().into();
        Ok(Authorization {
            verifying_key,
            signature,
        })
    }
}

impl Watchable<()> for Wallet {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the wallet changes
    fn watch(&self) -> Self::WatchStream {
        let Self {
            env: _,
            seed,
            address_to_index,
            epk_to_index,
            index_to_address,
            index_to_epk,
            index_to_vk,
            utxos,
            stxos,
            unconfirmed_utxos,
            spent_unconfirmed_utxos,
            truthcoin_reservations,
            known_truthcoin,
            vk_to_index,
            _version: _,
        } = self;
        let watchables = [
            seed.watch().clone(),
            address_to_index.watch().clone(),
            epk_to_index.watch().clone(),
            index_to_address.watch().clone(),
            index_to_epk.watch().clone(),
            index_to_vk.watch().clone(),
            utxos.watch().clone(),
            stxos.watch().clone(),
            unconfirmed_utxos.watch().clone(),
            spent_unconfirmed_utxos.watch().clone(),
            truthcoin_reservations.watch().clone(),
            known_truthcoin.watch().clone(),
            vk_to_index.watch().clone(),
        ];
        let streams = StreamMap::from_iter(
            watchables.into_iter().map(WatchStream::new).enumerate(),
        );
        let streams_len = streams.len();
        streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        })
    }
}
