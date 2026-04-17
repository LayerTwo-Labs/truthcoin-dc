use std::{
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
    types::{Bytes, SerdeBincode, U8, U32},
};
use libes::EciesError;
use serde::{Deserialize, Serialize};
use sneed::{DbError, Env, EnvError, RwTxnError, UnitKey, db, env, rwtxn};
use thiserror::Error;
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    authorization::{self, Authorization, Signature, get_address},
    math::{
        markets,
        safe_math::{Rounding, to_sats},
        trading,
    },
    state::markets::{
        DEFAULT_MARKET_BETA, DimensionSpec, MarketId, parse_dimensions,
    },
    types::{
        Address, AmountOverflowError, AmountUnderflowError,
        AuthorizedTransaction, BitcoinOutputContent, EncryptionPubKey,
        FilledOutput, GetBitcoinValue, InPoint, OutPoint, Output,
        OutputContent, SpentOutput, Transaction, TxData, VERSION, VerifyingKey,
        Version, WithdrawalOutputContent, keys::Ecies,
    },
    util::Watchable,
};

fn sorted_outpoints(utxos: HashMap<OutPoint, Output>) -> Vec<OutPoint> {
    let mut pairs: Vec<_> = utxos.into_iter().collect();
    pairs.sort_by_key(|(outpoint, _)| *outpoint);
    pairs.into_iter().map(|(outpoint, _)| outpoint).collect()
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
}

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
    #[error("Invalid decision ID: {reason}")]
    InvalidDecisionId { reason: String },
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
    /// Map each verifying key to it's index
    vk_to_index: DatabaseUnique<SerdeBincode<VerifyingKey>, U32<BigEndian>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 12;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            use heed::EnvFlags;
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS);
            // Wallet DB keeps fsync enabled (no NO_SYNC/NO_META_SYNC)
            // because the seed and key material cannot be
            // reconstructed from the chain. The DB is small (~10MB)
            // and low write volume, so fsync cost is negligible.
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
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

    pub fn voter_address(&self) -> Result<Address, Error> {
        let mut txn = self.env.write_txn()?;
        if let Some(addr) = self
            .index_to_address
            .try_get(&txn, &0)
            .map_err(DbError::from)?
        {
            txn.abort();
            return Ok(addr);
        }
        let tx_signing_key = self.get_tx_signing_key(&txn, 0)?;
        let address = get_address(&tx_signing_key.verifying_key().into());
        self.index_to_address.put(&mut txn, &0, &address)?;
        self.address_to_index.put(&mut txn, &address, &0)?;
        txn.commit()?;
        Ok(address)
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
        let seed_bytes: [u8; 64] =
            seed.as_bytes().try_into().map_err(|_| Error::NoSeed)?;
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
        let mut outputs = vec![Output::new(
            self.get_new_address()?,
            OutputContent::Withdrawal(WithdrawalOutputContent {
                value,
                main_fee,
                main_address,
            }),
        )];
        self.push_bitcoin_change(&mut outputs, change)?;
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
        self.push_bitcoin_change(&mut outputs, change)?;
        Ok(Transaction::new(inputs, outputs))
    }

    /// Select confirmed Bitcoin UTXOs only, following Bitcoin Hivemind's requirement
    /// that only confirmed UTXOs can be spent in block construction.
    /// This prevents the "utxo doesn't exist" error when trying to spend mempool UTXOs.
    pub fn select_bitcoins(
        &self,
        value: bitcoin::Amount,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;

        let mut bitcoin_utxos = Vec::with_capacity(64);

        let mut iter = self.utxos.iter(&rotxn).map_err(DbError::from)?;
        while let Some((outpoint, filled_output)) =
            iter.next().map_err(DbError::from)?
        {
            if filled_output.is_bitcoin()
                && !filled_output.content.is_withdrawal()
                && filled_output.get_bitcoin_value() > bitcoin::Amount::ZERO
            {
                let output: Output = filled_output.into();
                bitcoin_utxos.push((outpoint, output));
            }
        }

        bitcoin_utxos.sort_unstable_by_key(
            |(_, output): &(OutPoint, Output)| {
                std::cmp::Reverse(output.get_bitcoin_value())
            },
        );

        let mut selected = HashMap::with_capacity(bitcoin_utxos.len().min(10));
        let mut total = bitcoin::Amount::ZERO;

        for (outpoint, output) in &bitcoin_utxos {
            total = total
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(*outpoint, output.clone());

            if total >= value {
                return Ok((total, selected));
            }
        }

        Err(Error::NotEnoughFunds)
    }

    fn push_bitcoin_change(
        &self,
        outputs: &mut Vec<Output>,
        change: bitcoin::Amount,
    ) -> Result<(), Error> {
        if change > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ));
        }
        Ok(())
    }

    /// Create a transaction to claim a decision.
    /// Returns a new transaction ready to be signed and sent.
    pub fn claim_decision(
        &self,
        input: DecisionClaimInput,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let DecisionClaimInput {
            decision_type,
            decisions,
        } = input;

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change = total_bitcoin - fee;
        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();

        let mut outputs = Vec::new();
        self.push_bitcoin_change(&mut outputs, change)?;

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(TxData::ClaimDecision {
            decision_type,
            decisions,
        });

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
        let (total_bitcoin, bitcoin_utxos) =
            self.select_bitcoins(total_cost)?;
        let change = total_bitcoin - total_cost;

        // Collect inputs first so creator_address matches the first
        // transaction input, consistent with extract_creator_address
        // in block validation (which uses spent_utxos.first())
        let inputs: Vec<_> = bitcoin_utxos.keys().copied().collect();
        let creator_address = bitcoin_utxos
            .get(inputs.first().ok_or(Error::NotEnoughFunds)?)
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
            trading_fee,
            category_txids: None,
            residual_names: None,
            tx_pow_hash_selector,
            tx_pow_ordering,
            tx_pow_difficulty,
        };
        let mut outputs = Vec::new();

        // Create explicit MarketFunds (treasury) output with treasury funding
        if treasury_sats > 0 {
            let treasury_address = generate_market_treasury_address(&market_id);
            outputs.push(Output::new(
                treasury_address,
                OutputContent::MarketFunds {
                    market_id: market_id_bytes,
                    amount: BitcoinOutputContent(bitcoin::Amount::from_sat(
                        treasury_sats,
                    )),
                    is_fee: false,
                },
            ));
        }

        self.push_bitcoin_change(&mut outputs, change)?;

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok((tx, market_id))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn trade(
        &self,
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
            let (_total_bitcoin, bitcoin_utxos) =
                self.select_bitcoins(bitcoin::Amount::from_sat(limit_sats))?;
            sorted_outpoints(bitcoin_utxos)
        } else {
            let min_fee =
                bitcoin::Amount::from_sat(trading::TRADE_MINER_FEE_SATS);
            let (_total_bitcoin, bitcoin_utxos) =
                self.select_bitcoins(min_fee)?;
            sorted_outpoints(bitcoin_utxos)
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

        let mut tx = Transaction::new(inputs, outputs);
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

        let (_total_bitcoin, bitcoin_utxos) =
            self.select_bitcoins(total_needed)?;

        let has_author_input = bitcoin_utxos
            .values()
            .any(|output| output.address == market_author);
        if !has_author_input {
            return Err(Error::NotEnoughFunds);
        }

        let inputs = sorted_outpoints(bitcoin_utxos);

        let mut tx = Transaction::new(inputs, Vec::new());
        tx.data = Some(TxData::AmplifyBeta {
            market_id: MarketId::new(*market_id.as_bytes()),
            amount,
            market_author,
        });

        Ok(tx)
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

    /// Authorize a transaction with strict validation against mempool UTXO spending.
    /// Following Bitcoin Hivemind's requirement that only confirmed UTXOs can be spent.
    pub fn authorize(
        &self,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error> {
        let rotxn = self.env.read_txn()?;
        let mut authorizations = vec![];
        let mut input_addresses = std::collections::HashSet::new();

        for input in &transaction.inputs {
            let spent_utxo: Output = if let Some(filled_utxo) =
                self.utxos.try_get(&rotxn, input).map_err(DbError::from)?
            {
                filled_utxo.into()
            } else {
                if let Some(_unconfirmed_utxo) = self
                    .unconfirmed_utxos
                    .try_get(&rotxn, input)
                    .map_err(DbError::from)?
                {
                    tracing::error!(
                        "Attempted to spend unconfirmed UTXO {:?}",
                        input
                    );
                    return Err(Error::NoUtxo);
                }
                return Err(Error::NoUtxo);
            };

            input_addresses.insert(spent_utxo.address);
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

        let actor_proof =
            self.build_actor_proof(&rotxn, &transaction, &input_addresses)?;

        Ok(AuthorizedTransaction {
            authorizations,
            transaction,
            actor_proof,
        })
    }

    fn build_actor_proof(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
        input_addresses: &std::collections::HashSet<Address>,
    ) -> Result<Option<Authorization>, Error> {
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
                    self.get_tx_signing_key_for_addr(rotxn, &addr)?;
                let signature =
                    crate::authorization::sign_tx(&signing_key, transaction)?;
                Ok(Some(Authorization {
                    verifying_key: signing_key.verifying_key().into(),
                    signature,
                }))
            }
            None => Ok(None),
        }
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

    pub fn submit_ballot(
        &self,
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

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change_bitcoin = total_bitcoin - fee;

        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();
        let mut outputs = Vec::new();
        self.push_bitcoin_change(&mut outputs, change_bitcoin)?;

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
    }

    pub fn transfer_reputation(
        &self,
        dest: Address,
        amount: f64,
        fee: bitcoin::Amount,
        memo: Option<Vec<u8>>,
    ) -> Result<Transaction, Error> {
        let voter_addr = self.voter_address()?;
        let tx_data = crate::types::TransactionData::TransferReputation {
            sender: voter_addr,
            dest,
            amount,
        };

        let (total_bitcoin, bitcoin_utxos) =
            self.select_bitcoins_from_address(fee, voter_addr)?;
        let change_bitcoin = total_bitcoin - fee;

        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();
        let mut outputs = Vec::new();
        if change_bitcoin > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                voter_addr,
                OutputContent::Bitcoin(BitcoinOutputContent(change_bitcoin)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);
        tx.memo = memo.unwrap_or_default();

        Ok(tx)
    }

    fn select_bitcoins_from_address(
        &self,
        value: bitcoin::Amount,
        address: Address,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;

        let mut bitcoin_utxos = Vec::with_capacity(16);
        let mut iter = self.utxos.iter(&rotxn).map_err(DbError::from)?;
        while let Some((outpoint, filled_output)) =
            iter.next().map_err(DbError::from)?
        {
            if filled_output.address == address
                && filled_output.is_bitcoin()
                && !filled_output.content.is_withdrawal()
                && filled_output.get_bitcoin_value() > bitcoin::Amount::ZERO
            {
                let output: Output = filled_output.into();
                bitcoin_utxos.push((outpoint, output));
            }
        }

        bitcoin_utxos.sort_unstable_by_key(
            |(_, output): &(OutPoint, Output)| {
                std::cmp::Reverse(output.get_bitcoin_value())
            },
        );

        let mut selected = HashMap::with_capacity(bitcoin_utxos.len().min(10));
        let mut total = bitcoin::Amount::ZERO;
        for (outpoint, output) in &bitcoin_utxos {
            total = total
                .checked_add(output.get_bitcoin_value())
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
