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
        safe_math::{Rounding, to_sats},
        trading,
    },
    state::markets::{
        DEFAULT_MARKET_BETA, DimensionSpec, MarketId,
        generate_market_author_fee_address, generate_market_treasury_address,
        parse_dimensions,
    },
    types::{
        Address, AmountOverflowError, AmountUnderflowError, AssetId,
        AuthorizedTransaction, BitcoinOutputContent, EncryptionPubKey,
        FilledOutput, GetBitcoinValue, InPoint, OutPoint, Output,
        OutputContent, SpentOutput, Transaction, TxData, VERSION, VerifyingKey,
        Version, WithdrawalOutputContent, keys::Ecies,
    },
    util::Watchable,
};

/// Input struct for claiming a decision slot.
#[derive(Clone, Debug)]
pub struct SlotClaimInput {
    pub slot_id_bytes: [u8; 3],
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question: String,
    pub min: Option<i64>,
    pub max: Option<i64>,
}

/// Input struct for claiming multiple slots as a category.
/// The txid of the resulting transaction serves as the category identifier.
/// All category slots are binary (is_scaled=false) by design.
#[derive(Clone, Debug)]
pub struct CategoryClaimInput {
    /// List of (slot_id_bytes, question) pairs
    pub slots: Vec<([u8; 3], String)>,
    /// Whether these are standard slots
    pub is_standard: bool,
}

/// Input struct for creating a market.
///
/// Markets are defined using dimension bracket notation:
/// - Single binary decision: `[slot_id]`
/// - Multiple independent decisions: `[slot1,slot2,slot3]`
/// - Categorical (mutually exclusive): `[[slot1,slot2,slot3]]`
/// - Mixed dimensions: `[slot1,[slot2,slot3],slot4]`
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
    pub tags: Option<Vec<String>>,
    /// Initial liquidity in satoshis to fund the market (recommended).
    /// Beta is derived: β = liquidity / ln(num_outcomes)
    /// Mutually exclusive with beta - specify one or the other.
    pub initial_liquidity: Option<u64>,
    /// Txid(s) for categorical dimensions - required when using [[...]] notation
    /// Each txid must be a ClaimCategorySlots transaction
    pub category_txids: Option<Vec<[u8; 32]>>,
    /// Names for residual outcomes in categorical dimensions (one per [[...]])
    /// e.g., ["Bengals"] when explicit slots are Steelers/Ravens/Browns
    pub residual_names: Option<Vec<String>>,
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
    #[error("insufficient ownership proof for trader {trader}: {message}")]
    InsufficientOwnershipProof { trader: Address, message: String },
    #[error("utxo does not exist")]
    NoUtxo,
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[from] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error(transparent)]
    VkDoesNotExist(#[from] Box<VkDoesNotExistError>),
    #[error("Invalid slot ID: {reason}")]
    InvalidSlotId { reason: String },
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
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS);
            unsafe { Env::open(&env_open_options, path) }?
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

    pub fn create_votecoin_transfer(
        &self,
        address: Address,
        amount: u32,
        fee: bitcoin::Amount,
        memo: Option<Vec<u8>>,
    ) -> Result<Transaction, Error> {
        let (total_sats, bitcoins) = self.select_bitcoins(fee)?;
        let change_sats = total_sats - fee;
        let mut inputs: Vec<_> = bitcoins.into_keys().collect();
        let (total_votecoin, votecoin_utxos) =
            self.select_votecoin_utxos(amount)?;
        let votecoin_change = total_votecoin - amount;
        inputs.extend(votecoin_utxos.into_keys());
        let mut outputs = vec![Output {
            address,
            content: OutputContent::Votecoin(amount),
            memo: memo.unwrap_or_default(),
        }];
        if change_sats != Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change_sats)),
            ))
        }
        if votecoin_change != 0 {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Votecoin(votecoin_change),
            ))
        }
        let tx = Transaction::new(inputs, outputs);
        Ok(tx)
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
                && !filled_output.is_votecoin()
                && filled_output.get_bitcoin_value() > bitcoin::Amount::ZERO
            {
                let output: Output = filled_output.into();
                bitcoin_utxos.push((outpoint, output));
            }
        }

        bitcoin_utxos.sort_unstable_by_key(
            |(_, output): &(OutPoint, Output)| output.get_bitcoin_value(),
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

    /// Select Bitcoin UTXOs for sell transactions.
    /// Requires at least one UTXO from the trader address (to prove ownership)
    pub fn select_bitcoins_for_sell(
        &self,
        value: bitcoin::Amount,
        trader: Address,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;

        let mut trader_utxos = Vec::new();
        let mut other_utxos = Vec::new();

        let mut iter = self.utxos.iter(&rotxn).map_err(DbError::from)?;
        while let Some((outpoint, filled_output)) =
            iter.next().map_err(DbError::from)?
        {
            if filled_output.is_bitcoin()
                && !filled_output.content.is_withdrawal()
                && !filled_output.is_votecoin()
                && filled_output.get_bitcoin_value() > bitcoin::Amount::ZERO
            {
                let output: Output = filled_output.into();
                if output.address == trader {
                    trader_utxos.push((outpoint, output));
                } else {
                    other_utxos.push((outpoint, output));
                }
            }
        }

        // Must have at least one UTXO from trader to prove ownership
        if trader_utxos.is_empty() {
            return Err(Error::InsufficientOwnershipProof {
                trader,
                message: "Trader address has no Bitcoin UTXOs to prove ownership for sell. \
                         Please send some Bitcoin to this address first.".to_string(),
            });
        }

        // Sort by value (smallest first) for efficient selection
        trader_utxos
            .sort_unstable_by_key(|(_, output)| output.get_bitcoin_value());
        other_utxos
            .sort_unstable_by_key(|(_, output)| output.get_bitcoin_value());

        let mut selected = HashMap::with_capacity(16);
        let mut total = bitcoin::Amount::ZERO;

        // First, add trader's UTXOs (at least one required)
        for (outpoint, output) in trader_utxos {
            total = total
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(outpoint, output);

            if total >= value {
                return Ok((total, selected));
            }
        }

        // If trader's UTXOs aren't enough, add from other addresses
        for (outpoint, output) in other_utxos {
            total = total
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(outpoint, output);

            if total >= value {
                return Ok((total, selected));
            }
        }

        Err(Error::NotEnoughFunds)
    }

    /// Select confirmed Votecoin UTXOs only.
    pub fn select_votecoin_utxos(
        &self,
        value: u32,
    ) -> Result<(u32, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn()?;

        let mut votecoin_utxos = Vec::with_capacity(32);

        let mut iter = self.utxos.iter(&rotxn)?;
        while let Some((outpoint, filled_output)) = iter.next()? {
            if filled_output.is_votecoin()
                && !filled_output.content.is_withdrawal()
                && let Some(votecoin_value) = filled_output.votecoin()
            {
                let output: Output = filled_output.into();
                votecoin_utxos.push((outpoint, output, votecoin_value));
            }
        }

        votecoin_utxos
            .sort_unstable_by_key(|(_, _, votecoin_value)| *votecoin_value);

        let mut selected = HashMap::with_capacity(votecoin_utxos.len().min(8));
        let mut total_value: u32 = 0;

        for (outpoint, output, votecoin_value) in &votecoin_utxos {
            total_value = total_value
                .checked_add(*votecoin_value)
                .ok_or(Error::AmountOverflow(AmountOverflowError))?;
            selected.insert(*outpoint, output.clone());

            if total_value >= value {
                return Ok((total_value, selected));
            }
        }

        Err(Error::NotEnoughFunds)
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
            AssetId::Votecoin => {
                let (total, utxos) =
                    self.select_votecoin_utxos(amount.try_into().unwrap())?;
                Ok((total as u64, utxos))
            }
        }
    }

    // Select LP tokens with optimized collection and selection

    /// Create a transaction to claim a decision slot.
    /// Returns a new transaction ready to be signed and sent.
    pub fn claim_decision_slot(
        &self,
        input: SlotClaimInput,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let SlotClaimInput {
            slot_id_bytes,
            is_standard,
            is_scaled,
            question,
            min,
            max,
        } = input;

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change = total_bitcoin - fee;
        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();

        let mut outputs = Vec::new();

        if change > bitcoin::Amount::ZERO {
            let change_address = self.get_new_address()?;
            outputs.push(Output::new(
                change_address,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(TxData::ClaimDecisionSlot {
            slot_id_bytes,
            is_standard,
            is_scaled,
            question,
            min,
            max,
        });

        Ok(tx)
    }

    /// Create a transaction to claim multiple decision slots as a category.
    /// Returns a new transaction ready to be signed and sent.
    /// The txid of this transaction serves as the category identifier.
    pub fn claim_category_slots(
        &self,
        input: CategoryClaimInput,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let CategoryClaimInput { slots, is_standard } = input;

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change = total_bitcoin - fee;
        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();

        let mut outputs = Vec::new();

        if change > bitcoin::Amount::ZERO {
            let change_address = self.get_new_address()?;
            outputs.push(Output::new(
                change_address,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(TxData::ClaimCategorySlots { slots, is_standard });

        Ok(tx)
    }

    /// Estimate storage fee for dimensional market
    fn estimate_dimensional_storage_fee(
        &self,
        total_slots: usize,
        num_dimensions: usize,
    ) -> Result<bitcoin::Amount, Error> {
        // Dimensional markets have more complex outcome spaces
        // Base cost scales with slot count, bonus cost for multi-dimensional complexity
        let base_cost = (total_slots as u64) * 1000; // 1000 sats per slot
        let complexity_cost =
            (num_dimensions as u64) * (num_dimensions as u64) * 100; // Quadratic complexity cost

        let total_cost = base_cost + complexity_cost;
        Ok(bitcoin::Amount::from_sat(total_cost))
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
            tags,
            initial_liquidity,
            category_txids,
            residual_names,
        } = input;

        let dimension_specs = parse_dimensions(&dimensions).map_err(|_| {
            Error::InvalidSlotId {
                reason: "Failed to parse dimension specification".to_string(),
            }
        })?;

        // Validate residual_names count matches categorical dimension count
        if let Some(ref names) = residual_names {
            let categorical_count = dimension_specs
                .iter()
                .filter(|d| matches!(d, DimensionSpec::Categorical(_)))
                .count();
            if names.len() != categorical_count {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "residual_names count ({}) must match categorical dimension count ({})",
                        names.len(),
                        categorical_count
                    ),
                });
            }
        }

        let mut total_slots = 0;
        for spec in &dimension_specs {
            match spec {
                DimensionSpec::Single(_) => total_slots += 1,
                DimensionSpec::Categorical(slots) => total_slots += slots.len(),
            }
        }

        let storage_fee = self.estimate_dimensional_storage_fee(
            total_slots,
            dimension_specs.len(),
        )?;

        // Calculate number of resolved outcomes for beta derivation
        let num_outcomes: usize =
            dimension_specs.iter().fold(1, |acc, spec| {
                let spec_outcomes = match spec {
                    DimensionSpec::Single(_) => 2,
                    DimensionSpec::Categorical(slots) => slots.len() + 1,
                };
                acc * spec_outcomes
            });

        // Determine beta and treasury from inputs (mutually exclusive)
        // LMSR relationship: min_liquidity = β × ln(num_outcomes)
        let (beta, treasury_sats) = match (input_beta, initial_liquidity) {
            // Both provided: error - use one or the other
            (Some(_), Some(_)) => {
                return Err(Error::InvalidSlotId {
                    reason:
                        "Specify either beta or initial_liquidity, not both"
                            .to_string(),
                });
            }
            // Only beta (advanced): calculate minimum liquidity
            (Some(b), None) => {
                let liq_f64 =
                    trading::calculate_lmsr_liquidity(b, num_outcomes);
                let min_liq = to_sats(liq_f64, Rounding::Up)
                    .map_err(|_| AmountOverflowError)?;
                (b, min_liq)
            }
            // Only liquidity (primary user-facing): derive beta
            (None, Some(liq)) => {
                let derived_beta =
                    trading::derive_beta_from_liquidity(liq, num_outcomes);
                (derived_beta, liq)
            }
            // Neither: use defaults
            (None, None) => {
                let liq_f64 = trading::calculate_lmsr_liquidity(
                    DEFAULT_MARKET_BETA,
                    num_outcomes,
                );
                let min_liq = to_sats(liq_f64, Rounding::Up)
                    .map_err(|_| AmountOverflowError)?;
                (DEFAULT_MARKET_BETA, min_liq)
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

        // Get creator address from first selected UTXO
        let creator_address = bitcoin_utxos
            .values()
            .next()
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
            b: beta,
            trading_fee,
            tags,
            category_txids,
            residual_names,
        };

        let inputs: Vec<_> = bitcoin_utxos.into_keys().collect();
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

        // Add change output if needed
        if change > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok((tx, market_id))
    }

    /// # Arguments
    /// * `market_id` - The market to trade in
    /// * `outcome_index` - The outcome to trade shares for
    /// * `shares` - Number of shares to trade (positive = buy, negative = sell)
    /// * `trader` - The trader address (required for sell ownership validation)
    /// * `limit_sats` - Slippage limit: max_cost for buy, min_proceeds for sell
    /// * `base_sats` - LMSR base cost/proceeds (absolute)
    /// * `fee_sats` - Trading fee (absolute)
    /// * `tx_fee` - Transaction fee for the miner
    #[allow(clippy::too_many_arguments)]
    pub fn trade(
        &self,
        market_id: crate::state::markets::MarketId,
        outcome_index: usize,
        shares: f64,
        trader: Address,
        limit_sats: u64,
        base_sats: u64,
        fee_sats: u64,
        tx_fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let market_id_bytes = *market_id.as_bytes();
        let is_buy = shares > 0.0;

        let (inputs, outputs, change_address) = if is_buy {
            // Buy: need to pay base_sats + fee_sats to market + tx_fee
            let total_market_cost =
                bitcoin::Amount::from_sat(base_sats + fee_sats);
            let total_cost = tx_fee
                .checked_add(total_market_cost)
                .ok_or(AmountOverflowError)?;

            let (total_bitcoin, bitcoin_utxos) =
                self.select_bitcoins(total_cost)?;
            let change = total_bitcoin - total_cost;

            // Sort UTXOs for deterministic ordering
            let mut utxo_vec: Vec<_> = bitcoin_utxos.into_iter().collect();
            utxo_vec.sort_by_key(|(outpoint, _)| *outpoint);

            let inputs: Vec<_> =
                utxo_vec.into_iter().map(|(outpoint, _)| outpoint).collect();
            let mut outputs = Vec::new();

            // Treasury output (MarketFunds with is_fee=false)
            let treasury_address = generate_market_treasury_address(&market_id);
            outputs.push(Output::new(
                treasury_address,
                OutputContent::MarketFunds {
                    market_id: market_id_bytes,
                    amount: BitcoinOutputContent(bitcoin::Amount::from_sat(
                        base_sats,
                    )),
                    is_fee: false,
                },
            ));

            // Author fee output (MarketFunds with is_fee=true)
            let fee_address = generate_market_author_fee_address(&market_id);
            outputs.push(Output::new(
                fee_address,
                OutputContent::MarketFunds {
                    market_id: market_id_bytes,
                    amount: BitcoinOutputContent(bitcoin::Amount::from_sat(
                        fee_sats,
                    )),
                    is_fee: true,
                },
            ));

            // Change output goes to trader address (same as where shares are credited)
            // This ensures the address has Bitcoin for future sells
            if change > bitcoin::Amount::ZERO {
                outputs.push(Output::new(
                    trader,
                    OutputContent::Bitcoin(BitcoinOutputContent(change)),
                ));
            }

            (inputs, outputs, trader)
        } else {
            // Sell: only need tx_fee, payout comes from treasury during block connection.
            let (total_bitcoin, bitcoin_utxos) =
                self.select_bitcoins_for_sell(tx_fee, trader)?;
            let change = total_bitcoin - tx_fee;

            // Sort UTXOs for deterministic ordering, but keep trader's UTXOs first
            // to ensure validation sees ownership proof
            let mut trader_utxos: Vec<_> = bitcoin_utxos
                .iter()
                .filter(|(_, o)| o.address == trader)
                .map(|(op, o)| (*op, o.clone()))
                .collect();
            let mut other_utxos: Vec<_> = bitcoin_utxos
                .iter()
                .filter(|(_, o)| o.address != trader)
                .map(|(op, o)| (*op, o.clone()))
                .collect();
            trader_utxos.sort_by_key(|(outpoint, _)| *outpoint);
            other_utxos.sort_by_key(|(outpoint, _)| *outpoint);

            // Combine with trader UTXOs first
            let mut utxo_vec = trader_utxos;
            utxo_vec.extend(other_utxos);

            // Change goes back to trader address
            let inputs: Vec<_> =
                utxo_vec.into_iter().map(|(op, _)| op).collect();
            let mut outputs = Vec::new();

            // Only output is change (if any) - send to trader
            if change > bitcoin::Amount::ZERO {
                outputs.push(Output::new(
                    trader,
                    OutputContent::Bitcoin(BitcoinOutputContent(change)),
                ));
            }

            (inputs, outputs, trader)
        };

        let _ = change_address; // Used for logging if needed

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(TxData::Trade {
            market_id: MarketId::new(market_id_bytes),
            outcome_index: outcome_index as u32,
            shares,
            trader,
            limit_sats,
            base_sats,
            fee_sats,
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

    pub fn get_spent_unconfirmed_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, SpentOutput<OutputContent>>, Error> {
        let rotxn = self.env.read_txn()?;
        let stxos = self.spent_unconfirmed_utxos.iter(&rotxn)?.collect()?;
        Ok(stxos)
    }

    /// get all owned votecoin utxos
    pub fn get_votecoin(
        &self,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let mut utxos = self.get_utxos()?;
        utxos.retain(|_, output| output.is_votecoin());
        Ok(utxos)
    }

    /// get all spent votecoin utxos
    pub fn get_spent_votecoin(
        &self,
    ) -> Result<HashMap<OutPoint, SpentOutput>, Error> {
        let mut stxos = self.get_stxos()?;
        stxos.retain(|_, output| output.output.is_votecoin());
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

    /// Register as a voter
    pub fn register_voter(
        &self,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let tx_data = crate::types::TransactionData::RegisterVoter {
            initial_data: [0u8; 32],
        };

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change = total_bitcoin - fee;

        let inputs = bitcoin_utxos.into_keys().collect();
        let mut outputs = Vec::new();

        if change > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
    }

    /// Submit a single vote
    pub fn submit_vote(
        &self,
        slot_id_bytes: [u8; 3],
        vote_value: f64,
        voting_period: u32,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let tx_data = crate::types::TransactionData::SubmitVote {
            slot_id_bytes,
            vote_value,
            voting_period,
        };

        // First input must be Votecoin UTXO to verify voting rights
        let (total_votecoin, votecoin_utxos) = self.select_votecoin_utxos(1)?;
        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change_bitcoin = total_bitcoin - fee;

        let voter_address = votecoin_utxos
            .values()
            .next()
            .ok_or_else(|| Error::NotEnoughFunds)?
            .address;

        let mut inputs: Vec<_> = votecoin_utxos.into_keys().collect();
        inputs.extend(bitcoin_utxos.into_keys());

        let mut outputs = Vec::new();

        outputs.push(Output::new(
            voter_address,
            OutputContent::Votecoin(total_votecoin),
        ));

        if change_bitcoin > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change_bitcoin)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
    }

    /// Submit multiple votes in a batch
    pub fn submit_vote_batch(
        &self,
        votes: Vec<crate::types::VoteBatchItem>,
        voting_period: u32,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let tx_data = crate::types::TransactionData::SubmitVoteBatch {
            votes,
            voting_period,
        };

        // First input must be Votecoin UTXO to verify voting rights
        let (total_votecoin, votecoin_utxos) = self.select_votecoin_utxos(1)?;

        let (total_bitcoin, bitcoin_utxos) = self.select_bitcoins(fee)?;
        let change_bitcoin = total_bitcoin - fee;

        let voter_address = votecoin_utxos
            .values()
            .next()
            .ok_or_else(|| Error::NotEnoughFunds)?
            .address;

        let mut inputs: Vec<_> = votecoin_utxos.into_keys().collect();
        inputs.extend(bitcoin_utxos.into_keys());

        let mut outputs = Vec::new();

        outputs.push(Output::new(
            voter_address,
            OutputContent::Votecoin(total_votecoin),
        ));

        if change_bitcoin > bitcoin::Amount::ZERO {
            outputs.push(Output::new(
                self.get_new_address()?,
                OutputContent::Bitcoin(BitcoinOutputContent(change_bitcoin)),
            ));
        }

        let mut tx = Transaction::new(inputs, outputs);
        tx.data = Some(tx_data);

        Ok(tx)
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
