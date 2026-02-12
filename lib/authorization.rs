use std::str::FromStr;

use borsh::BorshSerialize;
use hex::FromHex;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction, Verify,
    VerifyingKey,
};

pub use ed25519_dalek::{SignatureError, Signer, SigningKey, Verifier};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ToSchema)]
#[repr(transparent)]
#[schema(value_type = String)]
pub struct Signature(pub ed25519_dalek::Signature);

impl BorshSerialize for Signature {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        self.0.to_bytes().serialize(writer)
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)
        } else {
            ed25519_dalek::Signature::deserialize(deserializer).map(Self)
        }
    }
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FromHex for Signature {
    type Error = <[u8; ed25519_dalek::Signature::BYTE_SIZE] as FromHex>::Error;

    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error> {
        let bytes =
            <[u8; ed25519_dalek::Signature::BYTE_SIZE] as FromHex>::from_hex(
                hex,
            )?;
        Ok(Self(ed25519_dalek::Signature::from_bytes(&bytes)))
    }
}

impl FromStr for Signature {
    type Err = <Self as FromHex>::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

impl Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            hex::serde::serialize(self.0.to_bytes(), serializer)
        } else {
            self.0.serialize(serializer)
        }
    }
}

/// Domain seperation tag for signing messages
#[derive(Clone, Copy, Debug, Deserialize, Serialize, ToSchema)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "clap", value(rename_all = "lower"))]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum Dst {
    Transaction = 0,
    /// Arbitrary, non-protocol messages
    Arbitrary = u8::MAX,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519_dalek error")]
    DalekError(#[from] SignatureError),
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: Address,
        hash_verifying_key: Address,
    },
    #[error("signature count mismatch: expected {expected}, got {actual}")]
    SignatureCountMismatch { expected: usize, actual: usize },
}

#[derive(
    BorshSerialize,
    Debug,
    Clone,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Authorization {
    #[schema(schema_with = <String as utoipa::PartialSchema>::schema)]
    pub verifying_key: VerifyingKey,
    pub signature: Signature,
}

impl GetAddress for Authorization {
    fn get_address(&self) -> Address {
        get_address(&self.verifying_key)
    }
}

impl Verify for Authorization {
    type Error = Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error> {
        verify_authorized_transaction(transaction)?;
        Ok(())
    }

    fn verify_body(body: &Body) -> Result<(), Self::Error> {
        verify_authorizations(body)?;
        Ok(())
    }
}

pub fn get_address(verifying_key: &VerifyingKey) -> Address {
    let mut hasher = blake3::Hasher::new();
    let mut reader = hasher.update(&verifying_key.to_bytes()).finalize_xof();
    let mut output: [u8; 20] = [0; 20];
    reader.fill(&mut output);
    Address(output)
}

struct Package<'a> {
    messages: Vec<&'a [u8]>,
    signatures: Vec<ed25519_dalek::Signature>,
    verifying_keys: Vec<ed25519_dalek::VerifyingKey>,
}

/// Canonical message to sign a tx
fn tx_msg_canonical(tx: &Transaction) -> borsh::io::Result<Vec<u8>> {
    let mut buf = vec![Dst::Transaction as u8];
    borsh::to_writer(&mut buf, tx)?;
    Ok(buf)
}

pub fn verify_authorized_transaction(
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    let tx_msg_canonical = tx_msg_canonical(&transaction.transaction)?;
    let messages: Vec<_> = std::iter::repeat_n(
        tx_msg_canonical.as_slice(),
        transaction.authorizations.len(),
    )
    .collect();
    let (verifying_keys, signatures): (
        Vec<ed25519_dalek::VerifyingKey>,
        Vec<ed25519_dalek::Signature>,
    ) = transaction
        .authorizations
        .iter()
        .map(
            |Authorization {
                 verifying_key,
                 signature,
             }| (verifying_key.0, signature.0),
        )
        .unzip();
    ed25519_dalek::verify_batch(&messages, &signatures, &verifying_keys)?;
    Ok(())
}

pub fn verify_authorizations(body: &Body) -> Result<(), Error> {
    let input_numbers = body
        .transactions
        .iter()
        .map(|transaction| transaction.inputs.len());
    let serialized_transactions: Vec<Vec<u8>> = body
        .transactions
        .par_iter()
        .map(tx_msg_canonical)
        .collect::<Result<_, _>>()?;
    let serialized_transactions =
        serialized_transactions.iter().map(Vec::as_slice);
    let messages = input_numbers.zip(serialized_transactions).flat_map(
        |(input_number, serialized_transaction)| {
            std::iter::repeat_n(serialized_transaction, input_number)
        },
    );

    let pairs = body.authorizations.iter().zip(messages).collect::<Vec<_>>();

    let num_threads = rayon::current_num_threads();
    let num_authorizations = body.authorizations.len();
    let package_size = num_authorizations / num_threads;
    let mut packages: Vec<Package> = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let mut package = Package {
            messages: Vec::with_capacity(package_size),
            signatures: Vec::with_capacity(package_size),
            verifying_keys: Vec::with_capacity(package_size),
        };
        for (authorization, message) in
            &pairs[i * package_size..(i + 1) * package_size]
        {
            package.messages.push(*message);
            package.signatures.push(authorization.signature.0);
            package.verifying_keys.push(authorization.verifying_key.0);
        }
        packages.push(package);
    }
    for (authorization, message) in &pairs[num_threads * package_size..] {
        packages[num_threads - 1].messages.push(*message);
        packages[num_threads - 1]
            .signatures
            .push(authorization.signature.0);
        packages[num_threads - 1]
            .verifying_keys
            .push(authorization.verifying_key.0);
    }
    let total_sigs: usize = packages.iter().map(|p| p.signatures.len()).sum();
    if total_sigs != body.authorizations.len() {
        return Err(Error::SignatureCountMismatch {
            expected: body.authorizations.len(),
            actual: total_sigs,
        });
    }
    packages
        .par_iter()
        .map(
            |Package {
                 messages,
                 signatures,
                 verifying_keys,
             }| {
                ed25519_dalek::verify_batch(
                    messages,
                    signatures,
                    verifying_keys,
                )
            },
        )
        .collect::<Result<(), SignatureError>>()?;
    Ok(())
}

/// Sign a message with DST prefix
pub fn sign(signing_key: &SigningKey, dst: Dst, msg: &[u8]) -> Signature {
    let msg_buf = [&[dst as u8], msg].concat();
    Signature(signing_key.sign(&msg_buf))
}

/// Verify a message with DST prefix
pub fn verify(
    signature: Signature,
    verifying_key: &VerifyingKey,
    dst: Dst,
    msg: &[u8],
) -> bool {
    let msg_buf = [&[dst as u8], msg].concat();
    verifying_key
        .0
        .verify_strict(&msg_buf, &signature.0)
        .is_ok()
}

pub fn sign_tx(
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error> {
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    Ok(sign(signing_key, Dst::Transaction, &tx_bytes_canonical))
}

pub fn authorize(
    addresses_signing_keys: &[(Address, &SigningKey)],
    transaction: Transaction,
) -> Result<AuthorizedTransaction, Error> {
    let mut authorizations: Vec<Authorization> =
        Vec::with_capacity(addresses_signing_keys.len());
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    for (address, signing_key) in addresses_signing_keys {
        let verifying_key = signing_key.verifying_key().into();
        let hash_verifying_key = get_address(&verifying_key);
        if *address != hash_verifying_key {
            return Err(Error::WrongKeyForAddress {
                address: *address,
                hash_verifying_key,
            });
        }
        let authorization = Authorization {
            verifying_key,
            signature: sign(signing_key, Dst::Transaction, &tx_bytes_canonical),
        };
        authorizations.push(authorization);
    }
    Ok(AuthorizedTransaction {
        authorizations,
        transaction,
    })
}
