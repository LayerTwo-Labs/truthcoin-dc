use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

use bip32ish::U31;
use byteorder::{BigEndian, ByteOrder};
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::types::{Bytes, SerdeBincode, U8, Unit};
use libes::EciesError;
use sneed::{Env, EnvError, RwTxnError, UnitKey, db::error::Error as DbError};
use thiserror::Error;
use tokio_stream::{StreamMap, wrappers::WatchStream};
use transitive::Transitive;

use crate::{
    math::{
        markets,
        safe_math::{Rounding, to_sats},
        trading,
    },
    state::markets::{
        DEFAULT_MARKET_BETA, DimensionSpec, MarketId, parse_dimensions,
    },
    types::{
        Accumulator, Address, AmountOverflowError, AmountUnderflowError,
        AuthorizedTransaction, EncryptionPubKey, GetValue, Hash, InPoint,
        OutPoint, OutPointKey, Output, OutputContent, PointedOutput,
        SpentOutput, THIS_SIDECHAIN, Transaction, TxData, UtreexoError,
        UtreexoNodeHash, VERSION, VerifyingKey, Version,
        authorization::{
            self, Authorization, Signature, SigningKey, get_address,
            rand_core::CryptoRng,
        },
        hash,
        keys::Ecies,
        wallet::Balance,
    },
    util::Watchable,
};

pub mod bip32;

/// Inputs that spend `coins`, each with its utxo hash
fn spend_inputs(coins: HashMap<OutPoint, Output>) -> Vec<(OutPoint, Hash)> {
    coins
        .into_iter()
        .map(|(outpoint, output)| {
            let utxo_hash = hash(&PointedOutput { outpoint, output });
            (outpoint, utxo_hash)
        })
        .collect()
}

/// A market transaction. The node adds the Utreexo proof on submit.
fn new_tx(inputs: Vec<(OutPoint, Hash)>, outputs: Vec<Output>) -> Transaction {
    Transaction {
        inputs: inputs.into(),
        proof: Default::default(),
        outputs: outputs.into(),
        data: None,
    }
}

/// Purpose (third path component) of each kind of wallet key
#[derive(Clone, Copy)]
#[repr(u32)]
enum KeyPurpose {
    TxSigning = 0,
    Encryption = 1,
    MessageSigning = 2,
}

#[derive(Clone, Debug)]
pub struct DecisionClaimInput {
    pub decision_type: crate::state::decisions::DecisionType,
    pub decisions: Vec<crate::types::DecisionClaimEntry>,
}

/// Input struct for creating a market.
///
/// Markets are defined using dimension bracket notation:
/// - Single binary decision: `[decision_id]`
/// - Multiple independent decisions: `[dec1,dec2,dec3]`
/// - Categorical (mutually exclusive): `[[dec1,dec2,dec3]]`
/// - Mixed dimensions: `[dec1,[dec2,dec3],dec4]`
///
/// `new_claims` lets the caller claim decisions in the same tx that
/// creates the market. Decision IDs referenced inside `dimensions` may
/// resolve to entries in `new_claims` instead of pre-existing chain state.
#[derive(Clone, Debug)]
pub struct CreateMarketInput {
    pub title: String,
    pub description: String,
    /// Dimension specification in bracket notation
    pub dimensions: String,
    /// Advanced: LMSR liquidity parameter controlling price sensitivity.
    /// Higher beta = more liquid = smaller price moves per trade.
    /// Mutually exclusive with initial_liquidity - specify one or the other.
    pub beta: Option<f64>,
    pub trading_fee: Option<f64>,
    /// Initial liquidity in satoshis to fund the market (recommended).
    /// Beta is derived: β = liquidity / ln(num_outcomes)
    /// Mutually exclusive with beta - specify one or the other.
    pub initial_liquidity: Option<u64>,
    pub category_option_counts: Option<Vec<usize>>,
    pub tx_pow_hash_selector: Option<u8>,
    pub tx_pow_ordering: Option<u8>,
    pub tx_pow_difficulty: Option<u8>,
    /// Decisions to claim inside this market creation tx. Empty for
    /// markets that reuse only already-claimed decisions.
    pub new_claims: Vec<crate::types::ClaimDecisionPayload>,
}

#[derive(Debug, Error)]
#[error("Message signature verification key {vk} does not exist")]
pub struct VkDoesNotExistError {
    vk: VerifyingKey,
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(
    from(bip32::HardenedDeriveError, bip32::Error),
    from(bip32::NonHardenedDeriveError, bip32::Error)
)]
pub enum Error {
    #[error("address {address} does not exist")]
    AddressDoesNotExist { address: crate::types::Address },
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("authorization error")]
    Authorization(#[from] crate::types::error::Authorization),
    #[error("bip32 error")]
    Bip32(#[from] bip32::Error),
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
    #[error(
        "Incompatible DB version ({}). Please clear the DB (`{}`) and re-sync",
        .version,
        .db_path.display()
    )]
    IncompatibleVersion { version: Version, db_path: PathBuf },
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("no index for address {address}")]
    NoIndex { address: Address },
    #[error(
        "wallet does not have a seed (set with RPC `set-seed-from-mnemonic`)"
    )]
    NoSeed,
    #[error("not enough funds")]
    NotEnoughFunds,
    #[error("no transfer destination")]
    NoTransferDestination,
    #[error("utxo does not exist")]
    NoUtxo,
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[source] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error(transparent)]
    VkDoesNotExist(#[from] Box<VkDoesNotExistError>),
    #[error("Invalid decision ID: {reason}")]
    InvalidDecisionId { reason: String },
}

/// Marker type for Wallet Env
pub struct WalletEnv;

type DatabaseUnique<KC, DC> = sneed::DatabaseUnique<KC, DC, WalletEnv>;
type RoTxn<'a> = sneed::RoTxn<'a, heed::AnyTls, WalletEnv>;

/// The coins a wallet picked for a transaction.
pub struct SelectedCoins {
    pub total: bitcoin::Amount,
    pub coins: HashMap<OutPoint, Output>,
    /// The picked coins that no block carries yet. The accumulator holds no
    /// leaf for them, so a proof leaves them out.
    pub unconfirmed: HashSet<OutPoint>,
}

#[derive(Clone)]
pub struct Wallet {
    env: sneed::Env<heed::WithoutTls, WalletEnv>,
    // Seed is always [u8; 64], but due to serde not implementing serialize
    // for [T; 64], use heed's `Bytes`
    // TODO: Don't store the seed in plaintext.
    seed: DatabaseUnique<U8, Bytes>,
    /// Map each address to it's index
    address_to_index:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<[u8; 4]>>,
    /// Map each address index to an address
    index_to_address:
        DatabaseUnique<SerdeBincode<[u8; 4]>, SerdeBincode<Address>>,
    utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    stxos: DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    /// Unconfirmed outputs that the wallet may spend. The node fills it from
    /// the mempool on every sync.
    unconfirmed_utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    /// Confirmed outputs that a mempool transaction already spends. Picking
    /// one again would make a double spend that the mempool refuses.
    mempool_spent_utxos: DatabaseUnique<OutPointKey, Unit>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
    /// Map each encryption pubkey to it's index
    epk_to_index:
        DatabaseUnique<SerdeBincode<EncryptionPubKey>, SerdeBincode<[u8; 4]>>,
    /// Map each encryption key index to an encryption pubkey
    index_to_epk:
        DatabaseUnique<SerdeBincode<[u8; 4]>, SerdeBincode<EncryptionPubKey>>,
    /// Map each verifying key to it's index
    vk_to_index:
        DatabaseUnique<SerdeBincode<VerifyingKey>, SerdeBincode<[u8; 4]>>,
    /// Map each message signing key index to a verifying key
    index_to_vk:
        DatabaseUnique<SerdeBincode<[u8; 4]>, SerdeBincode<VerifyingKey>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 12;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            use heed::EnvFlags;
            let mut env_open_options =
                heed::EnvOpenOptions::new().read_txn_without_tls();
            env_open_options
                // The wallet keeps every spent output, so a node that bids
                // for every mainchain block fills 10MB in weeks.
                .map_size(1024 * 1024 * 1024) // 1GB
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
            // WRITE_MAP/MAP_ASYNC/NO_READ_AHEAD are gated off on Windows:
            // LMDB's writable-mmap path returns ERROR_INVALID_HANDLE on commit
            // there.
            #[cfg(not(windows))]
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
                | EnvFlags::NO_SYNC
                | EnvFlags::NO_META_SYNC
                | EnvFlags::NO_READ_AHEAD;
            #[cfg(windows)]
            let fast_flags = EnvFlags::NO_SYNC | EnvFlags::NO_META_SYNC;
            unsafe { env_open_options.flags(fast_flags) };
            unsafe { Env::open(&env_open_options, path) }
                .map_err(EnvError::from)?
        };
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let seed_db = DatabaseUnique::create(&env, &mut rwtxn, "seed")
            .map_err(EnvError::from)?;
        let address_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "address_to_index")
                .map_err(EnvError::from)?;
        let index_to_address =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_address")
                .map_err(EnvError::from)?;
        let utxos = DatabaseUnique::create(&env, &mut rwtxn, "utxos")
            .map_err(EnvError::from)?;
        let stxos = DatabaseUnique::create(&env, &mut rwtxn, "stxos")
            .map_err(EnvError::from)?;
        let unconfirmed_utxos =
            DatabaseUnique::create(&env, &mut rwtxn, "unconfirmed_utxos")
                .map_err(EnvError::from)?;
        let mempool_spent_utxos =
            DatabaseUnique::create(&env, &mut rwtxn, "mempool_spent_utxos")
                .map_err(EnvError::from)?;
        let version = DatabaseUnique::create(&env, &mut rwtxn, "version")
            .map_err(EnvError::from)?;
        let epk_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "epk_to_index")
                .map_err(EnvError::from)?;
        let index_to_epk =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_epk")
                .map_err(EnvError::from)?;
        let vk_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "vk_to_index")
                .map_err(EnvError::from)?;
        let index_to_vk =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_vk")
                .map_err(EnvError::from)?;
        match version.try_get(&rwtxn, &()).map_err(DbError::from)? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 20,
                        patch: 0,
                    } =>
            {
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?,
        };
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            env,
            seed: seed_db,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
            unconfirmed_utxos,
            mempool_spent_utxos,
            _version: version,
            epk_to_index,
            index_to_epk,
            vk_to_index,
            index_to_vk,
        })
    }

    /// Overwrite the seed, or set it if it does not already exist.
    pub fn overwrite_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        self.seed.put(&mut rwtxn, &0, seed).map_err(DbError::from)?;
        self.address_to_index
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.index_to_address
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.utxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.stxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.mempool_spent_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn has_seed(&self) -> Result<bool, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self
            .seed
            .try_get(&rotxn, &0)
            .map_err(DbError::from)?
            .is_some())
    }

    /// Set the seed, if it does not already exist
    pub fn set_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
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
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let encryption_secret =
            self.get_encryption_secret_for_epk(&rotxn, encryption_pubkey)?;
        let res = Ecies::decrypt(&encryption_secret, ciphertext)
            .map_err(Error::Ecies)?;
        Ok(res)
    }

    pub fn create_withdrawal(
        &self,
        accumulator: &Accumulator,
        spend_zero_conf_change: bool,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        value: bitcoin::Amount,
        main_fee: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        tracing::trace!(
            accumulator = %accumulator.0,
            fee = %fee.display_dynamic(),
            ?main_address,
            main_fee = %main_fee.display_dynamic(),
            value = %value.display_dynamic(),
            "Creating withdrawal"
        );
        let selected = self.select_coins(
            value
                .checked_add(fee)
                .ok_or(AmountOverflowError)?
                .checked_add(main_fee)
                .ok_or(AmountOverflowError)?,
            spend_zero_conf_change,
        )?;
        let change = selected.total - value - fee - main_fee;

        let inputs: Vec<_> = selected
            .coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<UtreexoNodeHash> = inputs
            .iter()
            .filter(|(outpoint, _)| !selected.unconfirmed.contains(outpoint))
            .map(|(_, hash)| hash.into())
            .collect();
        let proof = accumulator.prove(&input_utxo_hashes)?;
        let outputs = vec![
            Output {
                address: self.get_new_address()?,
                content: OutputContent::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            },
            Output {
                address: self.get_new_address()?,
                content: OutputContent::Value(change),
            },
        ]
        .into();
        Ok(Transaction {
            inputs: inputs.into(),
            proof,
            outputs,
            data: None,
        })
    }

    pub fn create_transaction(
        &self,
        accumulator: &Accumulator,
        spend_zero_conf_change: bool,
        address: Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        self.create_transaction_many(
            accumulator,
            spend_zero_conf_change,
            &BTreeMap::from([(address, value)]),
            fee,
        )
    }

    /// Pay each address in `dests`, and pay the change to a new address
    pub fn create_transaction_many(
        &self,
        accumulator: &Accumulator,
        spend_zero_conf_change: bool,
        dests: &BTreeMap<Address, bitcoin::Amount>,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        if dests.is_empty() {
            return Err(Error::NoTransferDestination);
        }
        let value = dests
            .values()
            .try_fold(bitcoin::Amount::ZERO, |total, value| {
                total.checked_add(*value)
            })
            .ok_or(AmountOverflowError)?;
        let selected = self.select_coins(
            value.checked_add(fee).ok_or(AmountOverflowError)?,
            spend_zero_conf_change,
        )?;
        let change = selected.total - value - fee;
        let inputs: Vec<_> = selected
            .coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<UtreexoNodeHash> = inputs
            .iter()
            .filter(|(outpoint, _)| !selected.unconfirmed.contains(outpoint))
            .map(|(_, hash)| hash.into())
            .collect();
        let proof = accumulator.prove(&input_utxo_hashes)?;
        let mut outputs: Vec<Output> = dests
            .iter()
            .map(|(address, value)| Output {
                address: *address,
                content: OutputContent::Value(*value),
            })
            .collect();
        outputs.push(Output {
            address: self.get_new_address()?,
            content: OutputContent::Value(change),
        });
        let outputs = outputs.into();
        Ok(Transaction {
            inputs: inputs.into(),
            proof,
            outputs,
            data: None,
        })
    }

    /// Pick coins worth at least `value`. A confirmed coin comes first, so a
    /// chain of unconfirmed transactions only forms when the confirmed coins
    /// fall short. Bitcoin Core orders its coin selection the same way.
    ///
    /// `spend_zero_conf_change` decides whether the wallet's own unconfirmed
    /// change joins the pick at all.
    pub fn select_coins(
        &self,
        value: bitcoin::Amount,
        spend_zero_conf_change: bool,
    ) -> Result<SelectedCoins, Error> {
        use rayon::prelude::ParallelSliceMut;
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let mempool_spent: HashSet<OutPointKey> = self
            .mempool_spent_utxos
            .iter_keys(&rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        let mut utxos: Vec<_> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .collect::<Vec<_>>()
            .map_err(DbError::from)?;
        utxos.retain(|(outpoint_key, _)| !mempool_spent.contains(outpoint_key));
        utxos.par_sort_unstable_by_key(|(_, output)| output.get_value());
        let mut unconfirmed_utxos: Vec<_> = if spend_zero_conf_change {
            self.unconfirmed_utxos
                .iter(&rotxn)
                .map_err(DbError::from)?
                .collect()
                .map_err(DbError::from)?
        } else {
            Vec::new()
        };
        unconfirmed_utxos
            .par_sort_unstable_by_key(|(_, output)| output.get_value());
        let confirmed_count = utxos.len();

        let mut selected = HashMap::new();
        let mut unconfirmed = HashSet::new();
        let mut total = bitcoin::Amount::ZERO;
        for (index, (outpoint_key, output)) in
            utxos.iter().chain(&unconfirmed_utxos).enumerate()
        {
            if output.content.is_withdrawal()
                || output.content.is_market_funds()
            {
                continue;
            }
            if total > value {
                break;
            }
            total = total
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
            let outpoint: OutPoint = outpoint_key.into();
            selected.insert(outpoint, output.clone());
            if index >= confirmed_count {
                unconfirmed.insert(outpoint);
            }
        }
        if total < value {
            return Err(Error::NotEnoughFunds);
        }
        Ok(SelectedCoins {
            total,
            coins: selected,
            unconfirmed,
        })
    }

    pub fn delete_utxos(&self, outpoints: &[OutPoint]) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for outpoint in outpoints {
            let key = OutPointKey::from(outpoint);
            self.utxos.delete(&mut txn, &key).map_err(DbError::from)?;
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn spend_utxos(
        &self,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, inpoint) in spent {
            let key = OutPointKey::from(outpoint);
            let output =
                self.utxos.try_get(&txn, &key).map_err(DbError::from)?;
            if let Some(output) = output {
                self.utxos.delete(&mut txn, &key).map_err(DbError::from)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos
                    .put(&mut txn, &key, &spent_output)
                    .map_err(DbError::from)?;
            }
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    /// Make the confirmed table say what the chain says. `utxos` is every
    /// output the chain holds for this wallet, and `spent` is what a block
    /// spent since the last call.
    ///
    /// A block that disconnects takes an output off the chain without a spend,
    /// and it takes the utreexo leaf with it. A row that stays behind reads as
    /// confirmed, so `create_transaction` makes it a proof target and every
    /// send fails. Delete such a row here, where the chain's answer is known.
    pub fn sync_confirmed(
        &self,
        utxos: &HashMap<OutPoint, Output>,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, output) in utxos {
            self.utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), output)
                .map_err(DbError::from)?;
        }
        for (outpoint, inpoint) in spent {
            let key = OutPointKey::from(outpoint);
            let Some(output) =
                self.utxos.try_get(&rwtxn, &key).map_err(DbError::from)?
            else {
                continue;
            };
            self.utxos.delete(&mut rwtxn, &key).map_err(DbError::from)?;
            let spent_output = SpentOutput {
                output,
                inpoint: *inpoint,
            };
            self.stxos
                .put(&mut rwtxn, &key, &spent_output)
                .map_err(DbError::from)?;
        }
        let stale: Vec<OutPointKey> = self
            .utxos
            .iter_keys(&rwtxn)
            .map_err(DbError::from)?
            .filter(|key| Ok(!utxos.contains_key(&key.into())))
            .collect()
            .map_err(DbError::from)?;
        for key in &stale {
            self.utxos.delete(&mut rwtxn, key).map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn put_utxos(
        &self,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, output) in utxos {
            let key = OutPointKey::from(outpoint);
            self.utxos
                .put(&mut txn, &key, output)
                .map_err(DbError::from)?;
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    /// The value the wallet holds. A confirmed output that a mempool
    /// transaction already spends counts for nothing, because the money left.
    ///
    /// An unconfirmed output always counts toward `total` and `unconfirmed`,
    /// the way Bitcoin Core always reports such value. It counts toward
    /// `available` only when `spend_zero_conf_change` lets the wallet take it.
    pub fn get_balance(
        &self,
        spend_zero_conf_change: bool,
    ) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let () = self
            .utxos
            .iter(&txn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(key, utxo)| {
                if self
                    .mempool_spent_utxos
                    .try_get(&txn, &key)
                    .map_err(DbError::from)?
                    .is_some()
                {
                    return Ok(());
                }
                let value = utxo.get_value();
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
        let () = self
            .unconfirmed_utxos
            .iter(&txn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, utxo)| {
                let value = utxo.get_value();
                balance.total = balance
                    .total
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                balance.unconfirmed = balance
                    .unconfirmed
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                if spend_zero_conf_change && !utxo.content.is_withdrawal() {
                    balance.available = balance
                        .available
                        .checked_add(value)
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        Ok(balance)
    }

    /// Replace what the wallet knows about the mempool: the unconfirmed
    /// outputs it may spend, and the confirmed outputs a mempool transaction
    /// already spends. The node states both on every sync, so a wholesale
    /// replacement leaves no stale row behind when a transaction drops out.
    pub fn set_mempool_view(
        &self,
        unconfirmed: &HashMap<OutPoint, Output>,
        spent: &HashSet<OutPoint>,
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        self.unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.mempool_spent_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        for (outpoint, output) in unconfirmed {
            self.unconfirmed_utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), output)
                .map_err(DbError::from)?;
        }
        for outpoint in spent {
            self.mempool_spent_utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), &())
                .map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn get_mempool_spent_utxos(&self) -> Result<HashSet<OutPoint>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let outpoints: HashSet<OutPoint> = self
            .mempool_spent_utxos
            .iter_keys(&rotxn)
            .map_err(DbError::from)?
            .map(|key| Ok((&key).into()))
            .collect()
            .map_err(DbError::from)?;
        Ok(outpoints)
    }

    pub fn get_unconfirmed_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos: HashMap<OutPoint, Output> = self
            .unconfirmed_utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn get_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn get_stxos(&self) -> Result<HashMap<OutPoint, SpentOutput>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let stxos: HashMap<OutPoint, SpentOutput> = self
            .stxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(key, stxo)| Ok((key.into(), stxo)))
            .collect()
            .map_err(DbError::from)?;
        Ok(stxos)
    }

    pub fn get_addresses(&self) -> Result<HashSet<Address>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let addresses: HashSet<_> = self
            .index_to_address
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(_, address)| Ok(address))
            .collect()
            .map_err(DbError::from)?;
        Ok(addresses)
    }

    pub fn authorize<R>(
        &self,
        mut rng: R,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error>
    where
        R: CryptoRng,
    {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let mut authorizations = Vec::with_capacity(transaction.inputs.len());
        let mut input_addresses = HashSet::new();
        for (outpoint, _) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            let spent_utxo =
                match self.utxos.try_get(&txn, &key).map_err(DbError::from)? {
                    Some(spent_utxo) => spent_utxo,
                    None => self
                        .unconfirmed_utxos
                        .try_get(&txn, &key)
                        .map_err(DbError::from)?
                        .ok_or(Error::NoUtxo)?,
                };
            input_addresses.insert(spent_utxo.address);
            let index = self
                .address_to_index
                .try_get(&txn, &spent_utxo.address)
                .map_err(DbError::from)?
                .ok_or(Error::NoIndex {
                    address: spent_utxo.address,
                })?;
            let index = BigEndian::read_u32(&index);
            let signing_key = self.get_signing_key(&txn, index)?;
            let signature = crate::types::authorization::sign_tx(
                &mut rng,
                &signing_key,
                &transaction,
            )?;
            authorizations.push(Authorization {
                verifying_key: (&signing_key).into(),
                signature,
            });
        }
        let actor_proof = self
            .build_actor_proof(&mut rng, &txn, &transaction, &input_addresses)?
            .map(Box::new);
        Ok(AuthorizedTransaction {
            authorizations,
            transaction,
            actor_proof,
        })
    }

    fn build_actor_proof<R>(
        &self,
        rng: R,
        rotxn: &RoTxn,
        transaction: &Transaction,
        input_addresses: &std::collections::HashSet<Address>,
    ) -> Result<Option<Authorization>, Error>
    where
        R: CryptoRng,
    {
        use crate::types::TransactionData;

        let actor_addr = match &transaction.data {
            Some(TransactionData::Trade { trader, shares, .. })
                if *shares < 0 =>
            {
                if !input_addresses.contains(trader) {
                    Some(*trader)
                } else {
                    None
                }
            }
            Some(TransactionData::TransferReputation { sender, .. }) => {
                if !input_addresses.contains(sender) {
                    Some(*sender)
                } else {
                    None
                }
            }
            Some(TransactionData::SubmitVote { voter, .. })
            | Some(TransactionData::SubmitBallot { voter, .. }) => {
                if !input_addresses.contains(voter) {
                    Some(*voter)
                } else {
                    None
                }
            }
            _ => None,
        };

        match actor_addr {
            Some(addr) => {
                let signing_key =
                    self.get_signing_key_for_addr(rotxn, &addr)?;
                let signature = crate::types::authorization::sign_tx(
                    rng,
                    &signing_key,
                    transaction,
                )?;
                Ok(Some(Authorization {
                    verifying_key: (&signing_key).into(),
                    signature,
                }))
            }
            None => Ok(None),
        }
    }

    /// Derives an address the wallet never used. A change output takes one of
    /// these, so two transactions never share a change address.
    pub fn get_new_address(&self) -> Result<Address, Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let index =
            match self.index_to_address.last(&txn).map_err(DbError::from)? {
                Some((last_index, _)) => BigEndian::read_u32(&last_index) + 1,
                None => 0,
            };
        let signing_key = self.get_signing_key(&txn, index)?;
        let address = get_address(&(&signing_key).into());
        let index = index.to_be_bytes();
        self.index_to_address
            .put(&mut txn, &index, &address)
            .map_err(DbError::from)?;
        self.address_to_index
            .put(&mut txn, &address, &index)
            .map_err(DbError::from)?;
        txn.commit().map_err(RwTxnError::from)?;
        Ok(address)
    }

    /// The address to receive at. Derives a new one only once the current one
    /// receives.
    pub fn get_receive_address(&self) -> Result<Address, Error> {
        {
            let rotxn = self.env.read_txn().map_err(EnvError::from)?;
            let last =
                self.index_to_address.last(&rotxn).map_err(DbError::from)?;
            if let Some((_, address)) = last
                && !self.address_received(&rotxn, &address)?
            {
                return Ok(address);
            }
        }
        self.get_new_address()
    }

    /// True when any output the wallet holds or held pays this address.
    fn address_received(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<bool, Error> {
        let mut utxos = self.utxos.iter(rotxn).map_err(DbError::from)?;
        while let Some((_, output)) = utxos.next().map_err(DbError::from)? {
            if output.address == *address {
                return Ok(true);
            }
        }
        let mut stxos = self.stxos.iter(rotxn).map_err(DbError::from)?;
        while let Some((_, spent)) = stxos.next().map_err(DbError::from)? {
            if spent.output.address == *address {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Gets the latest generated address.
    pub fn try_get_last_address(&self) -> Result<Option<Address>, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let last = self.index_to_address.last(&txn).map_err(DbError::from)?;
        Ok(last.map(|(_, address)| address))
    }

    /// Gets the latest generated address, or generates a new one if no
    /// addresses have already been generated.
    pub fn get_or_generate_last_address(&self) -> Result<Address, Error> {
        if let Some(address) = self.try_get_last_address()? {
            Ok(address)
        } else {
            self.get_new_address()
        }
    }

    pub fn get_num_addresses(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let num = self.index_to_address.len(&txn).map_err(DbError::from)?;
        Ok(num as u32)
    }

    /// Derive the xpriv at
    /// m/43'/1899'/<purpose>'/<SIDECHAIN_NUMBER>'/0'/index
    /// (m / bip43 purpose / eCash Token / purpose / sidechain number /
    /// account / index)
    fn get_xpriv(
        &self,
        rotxn: &RoTxn,
        purpose: KeyPurpose,
        index: u32,
    ) -> Result<bip32::Xpriv, Error> {
        let seed = self
            .seed
            .try_get(rotxn, &0)
            .map_err(DbError::from)?
            .ok_or(Error::NoSeed)?;
        let mut xpriv = bip32::new_master_xpriv(seed);
        xpriv = xpriv.derive_hardened(U31::new(43).unwrap())?;
        xpriv = xpriv.derive_hardened(U31::new(1899).unwrap())?;
        xpriv = xpriv.derive_hardened(U31::new(purpose as u32).unwrap())?;
        xpriv =
            xpriv.derive_hardened(U31::new(THIS_SIDECHAIN as u32).unwrap())?;
        xpriv = xpriv.derive_hardened(U31::new(0).unwrap())?;
        match bip32ish::ChildIndex::from(index) {
            bip32ish::ChildIndex::Hardened { index } => {
                xpriv = xpriv.derive_hardened(index)?;
            }
            bip32ish::ChildIndex::NonHardened { index } => {
                xpriv = xpriv.derive_non_hardened(index)?;
            }
        }
        Ok(xpriv)
    }

    fn get_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<SigningKey, Error> {
        let xpriv = self.get_xpriv(rotxn, KeyPurpose::TxSigning, index)?;
        let sk = SigningKey::from_scalar(xpriv.secret_scalar)
            .expect("expected secret scalar to be non-zero");
        Ok(sk)
    }

    /// Get the tx signing key that corresponds to the provided address
    fn get_signing_key_for_addr(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<SigningKey, Error> {
        let index = self
            .address_to_index
            .try_get(rotxn, address)
            .map_err(DbError::from)?
            .ok_or(Error::AddressDoesNotExist { address: *address })?;
        let signing_key =
            self.get_signing_key(rotxn, BigEndian::read_u32(&index))?;
        // sanity check that signing key corresponds to address
        assert_eq!(*address, get_address(&(&signing_key).into()));
        Ok(signing_key)
    }

    fn get_encryption_secret(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<x25519_dalek::StaticSecret, Error> {
        let xpriv = self.get_xpriv(rotxn, KeyPurpose::Encryption, index)?;
        let secret = xpriv.secret_scalar.to_bytes().into();
        Ok(secret)
    }

    /// Get the encryption secret that corresponds to the provided encryption
    /// pubkey
    fn get_encryption_secret_for_epk(
        &self,
        rotxn: &RoTxn,
        epk: &EncryptionPubKey,
    ) -> Result<x25519_dalek::StaticSecret, Error> {
        let epk_idx = self
            .epk_to_index
            .try_get(rotxn, epk)
            .map_err(DbError::from)?
            .ok_or(Error::EpkDoesNotExist { epk: *epk })?;
        let encryption_secret =
            self.get_encryption_secret(rotxn, BigEndian::read_u32(&epk_idx))?;
        // sanity check that encryption secret corresponds to epk
        assert_eq!(*epk, (&encryption_secret).into());
        Ok(encryption_secret)
    }

    fn get_message_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<SigningKey, Error> {
        let xpriv = self.get_xpriv(rotxn, KeyPurpose::MessageSigning, index)?;
        let sk = SigningKey::from_scalar(xpriv.secret_scalar)
            .expect("expected secret scalar to be non-zero");
        Ok(sk)
    }

    /// Get the message signing key that corresponds to the provided
    /// verifying key
    fn get_message_signing_key_for_vk(
        &self,
        rotxn: &RoTxn,
        vk: &VerifyingKey,
    ) -> Result<SigningKey, Error> {
        let vk_idx = self
            .vk_to_index
            .try_get(rotxn, vk)
            .map_err(DbError::from)?
            .ok_or_else(|| Box::new(VkDoesNotExistError { vk: *vk }))?;
        let signing_key =
            self.get_message_signing_key(rotxn, BigEndian::read_u32(&vk_idx))?;
        // sanity check that signing key corresponds to vk
        assert_eq!(*vk, (&signing_key).into());
        Ok(signing_key)
    }

    /// The address that votes and holds reputation: address index 0
    pub fn voter_address(&self) -> Result<Address, Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let index = 0u32.to_be_bytes();
        if let Some(address) = self
            .index_to_address
            .try_get(&txn, &index)
            .map_err(DbError::from)?
        {
            txn.abort();
            return Ok(address);
        }
        let signing_key = self.get_signing_key(&txn, 0)?;
        let address = get_address(&(&signing_key).into());
        self.index_to_address
            .put(&mut txn, &index, &address)
            .map_err(DbError::from)?;
        self.address_to_index
            .put(&mut txn, &address, &index)
            .map_err(DbError::from)?;
        txn.commit().map_err(RwTxnError::from)?;
        Ok(address)
    }

    pub fn get_new_encryption_key(&self) -> Result<EncryptionPubKey, Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let index = match self.index_to_epk.last(&txn).map_err(DbError::from)? {
            Some((last_index, _)) => BigEndian::read_u32(&last_index) + 1,
            None => 0,
        };
        let encryption_secret = self.get_encryption_secret(&txn, index)?;
        let epk = (&encryption_secret).into();
        let index = index.to_be_bytes();
        self.index_to_epk
            .put(&mut txn, &index, &epk)
            .map_err(DbError::from)?;
        self.epk_to_index
            .put(&mut txn, &epk, &index)
            .map_err(DbError::from)?;
        txn.commit().map_err(RwTxnError::from)?;
        Ok(epk)
    }

    /// Get a new message verifying key
    pub fn get_new_verifying_key(&self) -> Result<VerifyingKey, Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let index = match self.index_to_vk.last(&txn).map_err(DbError::from)? {
            Some((last_index, _)) => BigEndian::read_u32(&last_index) + 1,
            None => 0,
        };
        let signing_key = self.get_message_signing_key(&txn, index)?;
        let vk = (&signing_key).into();
        let index = index.to_be_bytes();
        self.index_to_vk
            .put(&mut txn, &index, &vk)
            .map_err(DbError::from)?;
        self.vk_to_index
            .put(&mut txn, &vk, &index)
            .map_err(DbError::from)?;
        txn.commit().map_err(RwTxnError::from)?;
        Ok(vk)
    }

    pub fn sign_arbitrary_msg<R>(
        &self,
        rng: R,
        verifying_key: &VerifyingKey,
        msg: &str,
    ) -> Result<Signature, Error>
    where
        R: CryptoRng,
    {
        use authorization::{Dst, sign};
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let signing_key =
            self.get_message_signing_key_for_vk(&rotxn, verifying_key)?;
        let res = sign(rng, &signing_key, Dst::Arbitrary, msg.as_bytes());
        Ok(res)
    }

    pub fn sign_arbitrary_msg_as_addr<R>(
        &self,
        rng: R,
        address: &Address,
        msg: &str,
    ) -> Result<Authorization, Error>
    where
        R: CryptoRng,
    {
        use authorization::{Dst, sign};
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let signing_key = self.get_signing_key_for_addr(&rotxn, address)?;
        let signature = sign(rng, &signing_key, Dst::Arbitrary, msg.as_bytes());
        let verifying_key = (&signing_key).into();
        Ok(Authorization {
            verifying_key,
            signature,
        })
    }

    fn push_bitcoin_change(
        &self,
        outputs: &mut Vec<Output>,
        change: bitcoin::Amount,
    ) -> Result<(), Error> {
        if change > bitcoin::Amount::ZERO {
            outputs.push(Output {
                address: self.get_new_address()?,
                content: OutputContent::Value(change),
            });
        }
        Ok(())
    }

    /// Create a transaction to claim a decision.
    /// Returns a new transaction ready to be signed and sent.
    pub fn claim_decision(
        &self,
        spend_zero_conf_change: bool,
        input: DecisionClaimInput,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let DecisionClaimInput {
            decision_type,
            decisions,
        } = input;

        let selected = self.select_coins(fee, spend_zero_conf_change)?;
        let change = selected.total - fee;
        let inputs = spend_inputs(selected.coins);

        let mut outputs = Vec::new();
        self.push_bitcoin_change(&mut outputs, change)?;

        let mut tx = new_tx(inputs, outputs);
        tx.data =
            Some(TxData::ClaimDecision(crate::types::ClaimDecisionPayload {
                decision_type,
                decisions,
            }));

        Ok(tx)
    }

    /// Create a prediction market using dimension bracket notation
    ///
    /// Implements Bitcoin Hivemind Section 3.1 - Market Creation
    ///
    /// Dimension notation examples:
    /// - Single binary: `[004008]`
    /// - Multiple independent: `[004008,004009]`
    /// - Categorical: `[[004008,004009,00400a]]`
    /// - Mixed: `[004008,[004009,00400a]]`
    pub fn create_market(
        &self,
        spend_zero_conf_change: bool,
        input: CreateMarketInput,
        fee: bitcoin::Amount,
    ) -> Result<(Transaction, crate::state::markets::MarketId), Error> {
        use crate::state::markets::{
            compute_market_id, generate_market_treasury_address,
        };

        let CreateMarketInput {
            title,
            description,
            dimensions,
            beta: input_beta,
            trading_fee,
            initial_liquidity,
            category_option_counts,
            tx_pow_hash_selector,
            tx_pow_ordering,
            tx_pow_difficulty,
            new_claims,
        } = input;

        let dimension_specs = parse_dimensions(&dimensions).map_err(|_| {
            Error::InvalidDecisionId {
                reason: "Failed to parse dimension specification".to_string(),
            }
        })?;

        let num_outcomes: usize = {
            let mut cat_idx = 0usize;
            dimension_specs.iter().fold(1, |acc, spec| match spec {
                DimensionSpec::Single(_) => acc * 2,
                DimensionSpec::Categorical(_) => {
                    let n = category_option_counts
                        .as_ref()
                        .and_then(|c| c.get(cat_idx).copied())
                        .unwrap_or(2);
                    cat_idx += 1;
                    acc * n
                }
            })
        };

        let storage_fee = bitcoin::Amount::from_sat(
            markets::market_storage_fee(num_outcomes),
        );

        // Determine treasury from inputs (mutually exclusive).
        // Beta is derived from treasury: `beta = treasury / ln(num_outcomes)`.
        let treasury_sats = match (input_beta, initial_liquidity) {
            (Some(_), Some(_)) => {
                return Err(Error::InvalidDecisionId {
                    reason:
                        "Specify either beta or initial_liquidity, not both"
                            .to_string(),
                });
            }
            (Some(b), None) => {
                let liq_f64 =
                    trading::calculate_lmsr_liquidity(b, num_outcomes);
                to_sats(liq_f64, Rounding::Up)
                    .map_err(|_| AmountOverflowError)?
            }
            (None, Some(liq)) => liq,
            (None, None) => {
                let liq_f64 = trading::calculate_lmsr_liquidity(
                    DEFAULT_MARKET_BETA,
                    num_outcomes,
                );
                to_sats(liq_f64, Rounding::Up)
                    .map_err(|_| AmountOverflowError)?
            }
        };

        // Calculate total cost: fee + storage + treasury
        let mut total_cost =
            fee.checked_add(storage_fee).ok_or(AmountOverflowError)?;
        if treasury_sats > 0 {
            total_cost = total_cost
                .checked_add(bitcoin::Amount::from_sat(treasury_sats))
                .ok_or(AmountOverflowError)?;
        }

        // Select UTXOs - need to get creator_address from first UTXO
        let SelectedCoins {
            total: total_bitcoin,
            coins: bitcoin_utxos,
            unconfirmed: _,
        } = self.select_coins(total_cost, spend_zero_conf_change)?;
        let change = total_bitcoin - total_cost;

        // Collect inputs first so creator_address matches the first
        // transaction input, consistent with extract_creator_address
        // in block validation (which uses spent_utxos.first())
        let inputs = spend_inputs(bitcoin_utxos.clone());
        let (first_input, _) = inputs.first().ok_or(Error::NotEnoughFunds)?;
        let creator_address = bitcoin_utxos
            .get(first_input)
            .ok_or(Error::NotEnoughFunds)?
            .address;

        // Compute market_id deterministically from content
        let market_id = compute_market_id(
            &title,
            &description,
            &creator_address,
            &dimension_specs,
        );
        let market_id_bytes = *market_id.as_bytes();

        let tx_data = TxData::CreateMarket {
            title,
            description,
            dimension_specs,
            new_claims,
            trading_fee,
            tx_pow_hash_selector,
            tx_pow_ordering,
            tx_pow_difficulty,
        };
        let mut outputs = Vec::new();

        // Create explicit MarketFunds (treasury) output with treasury funding
        if treasury_sats > 0 {
            let treasury_address = generate_market_treasury_address(&market_id);
            outputs.push(Output {
                address: treasury_address,
                content: OutputContent::MarketFunds {
                    market_id: market_id_bytes,
                    amount: bitcoin::Amount::from_sat(treasury_sats),
                    is_fee: false,
                },
            });
        }

        self.push_bitcoin_change(&mut outputs, change)?;

        let mut tx = new_tx(inputs, outputs);
        tx.data = Some(tx_data);

        Ok((tx, market_id))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn trade(
        &self,
        spend_zero_conf_change: bool,
        market_id: crate::state::markets::MarketId,
        outcome_index: usize,
        shares: i64,
        trader: Address,
        limit_sats: u64,
        tx_pow_config: Option<crate::types::tx_pow::TxPowConfig>,
        prev_block_hash: crate::types::BlockHash,
    ) -> Result<Transaction, Error> {
        let is_buy = shares > 0;

        let inputs = if is_buy {
            let selected = self.select_coins(
                bitcoin::Amount::from_sat(limit_sats),
                spend_zero_conf_change,
            )?;
            spend_inputs(selected.coins)
        } else {
            let min_fee =
                bitcoin::Amount::from_sat(trading::TRADE_MINER_FEE_SATS);
            let selected =
                self.select_coins(min_fee, spend_zero_conf_change)?;
            spend_inputs(selected.coins)
        };

        let outputs = Vec::new();

        let tx_pow_nonce = match tx_pow_config {
            Some(config) if config.is_enabled() => {
                let pow_data = crate::types::tx_pow::serialize_trade_for_pow(
                    market_id.as_bytes(),
                    outcome_index as u32,
                    shares,
                    &trader,
                    limit_sats,
                    &prev_block_hash,
                );
                Some(config.mine(&pow_data))
            }
            _ => None,
        };

        let mut tx = new_tx(inputs, outputs);
        tx.data = Some(TxData::Trade {
            market_id: MarketId::new(*market_id.as_bytes()),
            outcome_index: outcome_index as u32,
            shares,
            trader,
            limit_sats,
            tx_pow_nonce,
            prev_block_hash,
        });

        Ok(tx)
    }

    /// Build an `AmplifyBeta` transaction that adds `amount` sats to the
    /// market's treasury, increasing its LMSR beta (liquidity depth).
    /// The wallet must own a UTXO belonging to `market_author` so the
    /// authorization can prove author identity.
    pub fn amplify_beta(
        &self,
        spend_zero_conf_change: bool,
        market_id: crate::state::markets::MarketId,
        amount: u64,
        market_author: Address,
    ) -> Result<Transaction, Error> {
        if amount == 0 {
            return Err(Error::InvalidDecisionId {
                reason: "AmplifyBeta amount must be positive".to_string(),
            });
        }

        let total_needed = bitcoin::Amount::from_sat(
            amount
                .checked_add(trading::TRADE_MINER_FEE_SATS)
                .ok_or(AmountOverflowError)?,
        );

        let bitcoin_utxos = self
            .select_coins(total_needed, spend_zero_conf_change)?
            .coins;

        let has_author_input = bitcoin_utxos
            .values()
            .any(|output| output.address == market_author);
        if !has_author_input {
            return Err(Error::NotEnoughFunds);
        }

        let inputs = spend_inputs(bitcoin_utxos);

        let mut tx = new_tx(inputs, Vec::new());
        tx.data = Some(TxData::AmplifyBeta {
            market_id: MarketId::new(*market_id.as_bytes()),
            amount,
            market_author,
        });

        Ok(tx)
    }

    pub fn submit_ballot(
        &self,
        spend_zero_conf_change: bool,
        votes: Vec<crate::types::BallotItem>,
        voting_period: u32,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let voter = self.voter_address()?;
        let tx_data = crate::types::TransactionData::SubmitBallot {
            voter,
            votes,
            voting_period,
        };

        let selected = self.select_coins(fee, spend_zero_conf_change)?;
        let change_bitcoin = selected.total - fee;

        let inputs = spend_inputs(selected.coins);
        let mut outputs = Vec::new();
        self.push_bitcoin_change(&mut outputs, change_bitcoin)?;

        let mut tx = new_tx(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
    }

    pub fn transfer_reputation(
        &self,
        spend_zero_conf_change: bool,
        dest: Address,
        amount: f64,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let voter_addr = self.voter_address()?;
        let tx_data = crate::types::TransactionData::TransferReputation {
            sender: voter_addr,
            dest,
            amount,
        };

        let (total_bitcoin, bitcoin_utxos) = self
            .select_bitcoins_from_address(
                fee,
                voter_addr,
                spend_zero_conf_change,
            )?;
        let change_bitcoin = total_bitcoin - fee;

        let inputs = spend_inputs(bitcoin_utxos);
        let mut outputs = Vec::new();
        if change_bitcoin > bitcoin::Amount::ZERO {
            outputs.push(Output {
                address: voter_addr,
                content: OutputContent::Value(change_bitcoin),
            });
        }

        let mut tx = new_tx(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
    }

    /// Pick coins of `address` worth at least `value`. A confirmed coin comes
    /// first, the same way as in `select_coins`.
    fn select_bitcoins_from_address(
        &self,
        value: bitcoin::Amount,
        address: Address,
        spend_zero_conf_change: bool,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let mempool_spent: HashSet<OutPointKey> = self
            .mempool_spent_utxos
            .iter_keys(&rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;

        let mut bitcoin_utxos = Vec::with_capacity(16);
        let mut iter = self.utxos.iter(&rotxn).map_err(DbError::from)?;
        while let Some((outpoint, output)) =
            iter.next().map_err(DbError::from)?
        {
            if output.address == address
                && output.content.is_value()
                && output.get_value() > bitcoin::Amount::ZERO
                && !mempool_spent.contains(&outpoint)
            {
                bitcoin_utxos.push((OutPoint::from(outpoint), output));
            }
        }

        bitcoin_utxos.sort_unstable_by_key(
            |(_, output): &(OutPoint, Output)| {
                std::cmp::Reverse(output.get_value())
            },
        );
        if spend_zero_conf_change {
            let mut unconfirmed_utxos = Vec::new();
            let mut iter =
                self.unconfirmed_utxos.iter(&rotxn).map_err(DbError::from)?;
            while let Some((outpoint, output)) =
                iter.next().map_err(DbError::from)?
            {
                if output.address == address
                    && output.content.is_value()
                    && output.get_value() > bitcoin::Amount::ZERO
                {
                    unconfirmed_utxos.push((OutPoint::from(outpoint), output));
                }
            }
            unconfirmed_utxos.sort_unstable_by_key(
                |(_, output): &(OutPoint, Output)| {
                    std::cmp::Reverse(output.get_value())
                },
            );
            bitcoin_utxos.extend(unconfirmed_utxos);
        }

        let mut selected = HashMap::with_capacity(bitcoin_utxos.len().min(10));
        let mut total = bitcoin::Amount::ZERO;
        for (outpoint, output) in &bitcoin_utxos {
            total = total
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(*outpoint, output.clone());
            if total >= value {
                return Ok((total, selected));
            }
        }

        Err(Error::NotEnoughFunds)
    }
}

impl Watchable<()> for Wallet {
    type WatchStream = std::pin::Pin<Box<dyn Stream<Item = ()> + Send>>;

    /// Get a signal that notifies whenever the wallet changes
    fn watch(&self) -> Self::WatchStream {
        let Self {
            env: _,
            seed,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
            unconfirmed_utxos,
            mempool_spent_utxos,
            _version: _,
            epk_to_index,
            index_to_epk,
            vk_to_index,
            index_to_vk,
        } = self;
        let watchables = [
            seed.watch().clone(),
            address_to_index.watch().clone(),
            index_to_address.watch().clone(),
            utxos.watch().clone(),
            stxos.watch().clone(),
            unconfirmed_utxos.watch().clone(),
            mempool_spent_utxos.watch().clone(),
            epk_to_index.watch().clone(),
            index_to_epk.watch().clone(),
            vk_to_index.watch().clone(),
            index_to_vk.watch().clone(),
        ];
        let streams = StreamMap::from_iter(
            watchables.into_iter().map(WatchStream::new).enumerate(),
        );
        let streams_len = streams.len();
        Box::pin(streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wallet_opens_a_database_it_created() -> anyhow::Result<()> {
        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "wallet-reopen-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            std::process::id()
        ))?;
        drop(Wallet::new(temp_dir.path())?);
        Wallet::new(temp_dir.path())?;
        Ok(())
    }

    /// A block that disconnects takes an output off the chain without a
    /// spend, and it takes the utreexo leaf with it. The confirmed row must go
    /// too, so the output moves to the unconfirmed side, where the proof code
    /// leaves it out.
    #[test]
    fn a_disconnected_output_leaves_the_confirmed_table() -> anyhow::Result<()>
    {
        use crate::types::OutputContent;

        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "wallet-disconnect-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            std::process::id()
        ))?;
        let wallet = Wallet::new(temp_dir.path())?;
        wallet.set_seed(&[0x99; 64])?;
        let address = wallet.get_new_address()?;
        let outpoint = OutPoint::Regular {
            txid: crate::types::hash(&[0u8; 32]).into(),
            vout: 0,
        };
        let output = Output {
            address,
            content: OutputContent::Value(bitcoin::Amount::from_sat(10_000)),
        };
        let held = HashMap::from([(outpoint, output.clone())]);

        wallet.sync_confirmed(&held, &[])?;
        anyhow::ensure!(wallet.get_utxos()?.len() == 1);

        // The chain drops the output, and no block spends it.
        wallet.sync_confirmed(&HashMap::new(), &[])?;
        anyhow::ensure!(
            wallet.get_utxos()?.is_empty(),
            "the confirmed row must go when the chain drops the output",
        );

        // The transaction that made it sits in the mempool again.
        wallet.set_mempool_view(&held, &HashSet::new())?;
        let balance = wallet.get_balance(true)?;
        anyhow::ensure!(
            balance.unconfirmed == bitcoin::Amount::from_sat(10_000),
            "the output reads as unconfirmed, got {balance:?}",
        );
        anyhow::ensure!(
            balance.total == bitcoin::Amount::from_sat(10_000),
            "the wallet counts the output one time, got {balance:?}",
        );
        anyhow::ensure!(
            balance.available == bitcoin::Amount::from_sat(10_000),
            "the wallet may take it, got {balance:?}",
        );

        // With the option off the value still shows, and the wallet may not
        // take it. Bitcoin Core reports such value the same way.
        let balance = wallet.get_balance(false)?;
        anyhow::ensure!(
            balance.unconfirmed == bitcoin::Amount::from_sat(10_000)
                && balance.total == bitcoin::Amount::from_sat(10_000),
            "the value stays visible, got {balance:?}",
        );
        anyhow::ensure!(
            balance.available == bitcoin::Amount::ZERO,
            "the wallet may not take it, got {balance:?}",
        );
        anyhow::ensure!(
            wallet
                .select_coins(bitcoin::Amount::from_sat(1_000), false)
                .is_err(),
            "coin selection must refuse the unconfirmed coin",
        );

        // With the option on the wallet takes it, and marks it unconfirmed so
        // the proof leaves it out.
        let selected =
            wallet.select_coins(bitcoin::Amount::from_sat(1_000), true)?;
        anyhow::ensure!(
            selected.unconfirmed.contains(&outpoint),
            "the wallet takes the coin and marks it unconfirmed",
        );
        Ok(())
    }

    #[test]
    fn test_get_receive_address() -> anyhow::Result<()> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let test_dir = std::env::temp_dir()
            .join(format!("truthcoin_dc_test_receive_{nanos}"));
        if test_dir.exists() {
            let _unused = std::fs::remove_dir_all(&test_dir);
        }

        let wallet = Wallet::new(&test_dir)?;
        wallet.set_seed(&[1u8; 64])?;

        // An address that never received comes back every time.
        let first = wallet.get_receive_address()?;
        for _ in 0..10 {
            assert_eq!(wallet.get_receive_address()?, first);
        }
        assert_eq!(wallet.get_addresses()?.len(), 1);

        // A fresh address is still fresh, so a change output never reuses one.
        let fresh = wallet.get_new_address()?;
        assert_ne!(fresh, first);
        assert_eq!(wallet.get_addresses()?.len(), 2);

        // The receive address moves on once it receives.
        let outpoint = OutPoint::Regular {
            txid: [0; 32].into(),
            vout: 0,
        };
        let output = Output {
            address: wallet.get_receive_address()?,
            content: OutputContent::Value(bitcoin::Amount::from_sat(1000)),
        };
        wallet.put_utxos(&HashMap::from([(outpoint, output)]))?;
        let second = wallet.get_receive_address()?;
        assert_ne!(second, first);
        assert_eq!(wallet.get_receive_address()?, second);

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_get_or_generate_last_address() -> anyhow::Result<()> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let test_dir = std::env::temp_dir()
            .join(format!("truthcoin_dc_test_wallet_{nanos}"));

        // Ensure clean state
        if test_dir.exists() {
            let _unused = std::fs::remove_dir_all(&test_dir);
        }

        let wallet = Wallet::new(&test_dir)?;

        assert!(!wallet.has_seed()?);
        assert!(wallet.try_get_last_address()?.is_none());
        let seed = [1u8; 64];
        wallet.set_seed(&seed)?;
        assert!(wallet.has_seed()?);

        assert!(wallet.try_get_last_address()?.is_none());
        let addr1 = wallet.get_or_generate_last_address()?;

        let last = wallet.try_get_last_address()?;
        assert_eq!(last, Some(addr1));

        let addr2 = wallet.get_or_generate_last_address()?;
        assert_eq!(addr1, addr2);

        let addr3 = wallet.get_new_address()?;
        assert_ne!(addr1, addr3);

        let last = wallet.try_get_last_address()?;
        assert_eq!(last, Some(addr3));

        let addr4 = wallet.get_or_generate_last_address()?;
        assert_eq!(addr3, addr4);

        // Clean up
        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    fn funded_wallet(
        name: &str,
        values_sats: &[u64],
    ) -> anyhow::Result<(std::path::PathBuf, Wallet, Accumulator)> {
        use crate::types::AccumulatorDiff;

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let test_dir = std::env::temp_dir()
            .join(format!("truthcoin_dc_test_{name}_{nanos}"));
        if test_dir.exists() {
            let _unused = std::fs::remove_dir_all(&test_dir);
        }
        let wallet = Wallet::new(&test_dir)?;
        wallet.set_seed(&[2u8; 64])?;

        let mut utxos = HashMap::new();
        let mut diff = AccumulatorDiff::default();
        for (index, value_sats) in values_sats.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid: [index as u8; 32].into(),
                vout: 0,
            };
            let output = Output {
                address: wallet.get_new_address()?,
                content: OutputContent::Value(bitcoin::Amount::from_sat(
                    *value_sats,
                )),
            };
            let pointed = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            diff.insert((&pointed).into());
            utxos.insert(outpoint, output);
        }
        wallet.put_utxos(&utxos)?;
        let mut accumulator = Accumulator::default();
        accumulator.apply_diff(diff)?;
        Ok((test_dir, wallet, accumulator))
    }

    fn value_of(output: &Output) -> u64 {
        output.get_value().to_sat()
    }

    #[test]
    fn test_create_transaction_many_pays_each_address() -> anyhow::Result<()> {
        let (test_dir, wallet, accumulator) =
            funded_wallet("transfer_many", &[10_000])?;

        let dests = BTreeMap::from([
            (Address([1u8; 20]), bitcoin::Amount::from_sat(1000)),
            (Address([2u8; 20]), bitcoin::Amount::from_sat(2000)),
            (Address([3u8; 20]), bitcoin::Amount::from_sat(3000)),
        ]);
        let fee = bitcoin::Amount::from_sat(500);
        let tx =
            wallet.create_transaction_many(&accumulator, true, &dests, fee)?;

        let outputs = tx.outputs.as_slice();
        assert_eq!(outputs.len(), 4);
        for (index, (address, value)) in dests.iter().enumerate() {
            assert_eq!(outputs[index].address, *address);
            assert_eq!(value_of(&outputs[index]), value.to_sat());
        }
        let change = &outputs[3];
        assert_eq!(value_of(change), 10_000 - 1000 - 2000 - 3000 - 500);
        assert!(wallet.get_addresses()?.contains(&change.address));

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    /// The node marks the inputs of a mempool transaction as mempool-spent.
    /// A second reputation transfer before a block must take another coin.
    #[test]
    fn a_second_reputation_transfer_skips_mempool_spent_coins()
    -> anyhow::Result<()> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let test_dir = std::env::temp_dir()
            .join(format!("truthcoin_dc_test_reputation_twice_{nanos}"));
        let wallet = Wallet::new(&test_dir)?;
        wallet.set_seed(&[3u8; 64])?;
        let voter = wallet.voter_address()?;
        let utxos: HashMap<OutPoint, Output> = [5_000, 4_000]
            .into_iter()
            .enumerate()
            .map(|(index, sats)| {
                let outpoint = OutPoint::Regular {
                    txid: [index as u8; 32].into(),
                    vout: 0,
                };
                let output = Output {
                    address: voter,
                    content: OutputContent::Value(bitcoin::Amount::from_sat(
                        sats,
                    )),
                };
                (outpoint, output)
            })
            .collect();
        wallet.put_utxos(&utxos)?;
        let dest = Address([5u8; 20]);
        let fee = bitcoin::Amount::from_sat(1_000);

        let first = wallet.transfer_reputation(true, dest, 0.1, fee)?;
        let first_inputs: HashSet<OutPoint> =
            first.inputs.iter().map(|(outpoint, _)| *outpoint).collect();
        wallet.set_mempool_view(&HashMap::new(), &first_inputs)?;
        let second = wallet.transfer_reputation(true, dest, 0.1, fee)?;
        let second_inputs: HashSet<OutPoint> = second
            .inputs
            .iter()
            .map(|(outpoint, _)| *outpoint)
            .collect();
        anyhow::ensure!(
            first_inputs.is_disjoint(&second_inputs),
            "both transfers spend {:?}",
            first_inputs
                .intersection(&second_inputs)
                .collect::<Vec<_>>(),
        );

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_create_transaction_keeps_one_payment_and_change()
    -> anyhow::Result<()> {
        let (test_dir, wallet, accumulator) =
            funded_wallet("transfer_one", &[10_000])?;

        let dest = Address([4u8; 20]);
        let tx = wallet.create_transaction(
            &accumulator,
            true,
            dest,
            bitcoin::Amount::from_sat(1000),
            bitcoin::Amount::from_sat(500),
        )?;

        let outputs = tx.outputs.as_slice();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].address, dest);
        assert_eq!(value_of(&outputs[0]), 1000);
        assert_eq!(value_of(&outputs[1]), 10_000 - 1000 - 500);
        assert!(wallet.get_addresses()?.contains(&outputs[1].address));

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_create_transaction_many_rejects_an_overflow() -> anyhow::Result<()>
    {
        let (test_dir, wallet, accumulator) =
            funded_wallet("transfer_overflow", &[10_000])?;

        let half = bitcoin::Amount::from_sat(bitcoin::Amount::MAX.to_sat() / 2);
        let dests = BTreeMap::from([
            (Address([1u8; 20]), half),
            (Address([2u8; 20]), half + bitcoin::Amount::from_sat(1)),
        ]);
        let result = wallet.create_transaction_many(
            &accumulator,
            true,
            &dests,
            bitcoin::Amount::from_sat(500),
        );
        assert!(matches!(result, Err(Error::AmountOverflow(_))));

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_create_transaction_many_needs_a_destination() -> anyhow::Result<()>
    {
        let (test_dir, wallet, accumulator) =
            funded_wallet("transfer_none", &[10_000])?;

        let result = wallet.create_transaction_many(
            &accumulator,
            true,
            &BTreeMap::new(),
            bitcoin::Amount::from_sat(500),
        );
        assert!(matches!(result, Err(Error::NoTransferDestination)));

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_create_transaction_many_totals_the_values() -> anyhow::Result<()> {
        let (test_dir, wallet, accumulator) =
            funded_wallet("transfer_total", &[1000, 1000])?;

        let dests = BTreeMap::from([
            (Address([1u8; 20]), bitcoin::Amount::from_sat(900)),
            (Address([2u8; 20]), bitcoin::Amount::from_sat(900)),
        ]);
        // Each coin alone is too small, so the sum decides the selection.
        let tx = wallet.create_transaction_many(
            &accumulator,
            true,
            &dests,
            bitcoin::Amount::from_sat(100),
        )?;
        assert_eq!(tx.inputs.len(), 2);
        assert_eq!(value_of(&tx.outputs.as_slice()[2]), 100);

        let result = wallet.create_transaction_many(
            &accumulator,
            true,
            &dests,
            bitcoin::Amount::from_sat(1000),
        );
        assert!(matches!(result, Err(Error::NotEnoughFunds)));

        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }
}
