use std::str::FromStr;

use borsh::BorshSerialize;
use const_hex::FromHex;
use rayon::{
    iter::{IntoParallelRefIterator as _, ParallelIterator as _},
    slice::ParallelSlice as _,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction, VerifyingKey,
};

pub use frost_ristretto255::rand_core;

pub type SignatureError = frost_ristretto255::Error;
pub type SigningKey = frost_ristretto255::SigningKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ToSchema)]
#[repr(transparent)]
#[schema(value_type = String)]
pub struct Signature(pub frost_ristretto255::Signature);

impl Signature {
    /// A compressed Ristretto point, then a scalar.
    pub const BYTE_SIZE: usize = 64;

    pub fn to_bytes(self) -> [u8; Self::BYTE_SIZE] {
        let mut out = [0u8; Self::BYTE_SIZE];
        out[..32].copy_from_slice(self.0.R().compress().as_bytes());
        out[32..].copy_from_slice(self.0.z().as_bytes());
        out
    }
}

impl BorshSerialize for Signature {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        self.to_bytes().serialize(writer)
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            const_hex::serde::deserialize(deserializer)
        } else {
            <frost_ristretto255::Signature as Deserialize>::deserialize(
                deserializer,
            )
            .map(Self)
        }
    }
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const_hex::encode(self.to_bytes()).fmt(f)
    }
}

impl FromHex for Signature {
    type Error = const_hex::FromHexError;

    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error> {
        let bytes = <[u8; Self::BYTE_SIZE] as FromHex>::from_hex(hex)?;
        frost_ristretto255::Signature::deserialize(&bytes)
            .map(Self)
            .map_err(|_| const_hex::FromHexError::InvalidStringLength)
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
            const_hex::serde::serialize(self.to_bytes(), serializer)
        } else {
            Serialize::serialize(&self.0, serializer)
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
    #[error("signature verification error")]
    SignatureVerification(#[from] SignatureError),
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

pub fn verify_actor_proof(
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    let actor_proof = match &transaction.actor_proof {
        Some(proof) => proof,
        None => return Ok(()),
    };
    let tx_msg_canonical = tx_msg_canonical(&transaction.transaction)?;
    actor_proof
        .verifying_key
        .0
        .verify(&tx_msg_canonical, &actor_proof.signature.0)?;
    Ok(())
}

pub fn verify_transaction(
    ctxt: &BatchVerificationContext,
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    verify_authorized_transaction(ctxt, transaction)?;
    verify_actor_proof(transaction)?;
    Ok(())
}

pub fn verify_body(
    ctxt: &BatchVerificationContext,
    body: &Body,
) -> Result<(), Error> {
    verify_authorizations(ctxt, body)?;
    let authorized_txs = body.authorized_transactions().map_err(|_| {
        Error::SignatureCountMismatch {
            expected: 0,
            actual: 0,
        }
    })?;
    for tx in &authorized_txs {
        verify_actor_proof(tx)?;
    }
    Ok(())
}

/// Derives a CSPRNG seed for a single batch verification.
struct BatchVerifier {
    hasher: blake3::Hasher,
    inner: frost_core::batch::Verifier<frost_ristretto255::Ristretto255Sha512>,
    /// Item counter, added as a suffix to the hasher before verification
    items: usize,
}

impl BatchVerifier {
    fn queue_item<Msg>(
        mut self,
        verifying_key: VerifyingKey,
        signature: Signature,
        msg: Msg,
    ) -> Result<Self, SignatureError>
    where
        Msg: AsRef<[u8]>,
    {
        let Self {
            inner,
            items,
            hasher,
        } = &mut self;
        let msg_bytes = msg.as_ref();
        hasher.update(&verifying_key.to_bytes());
        hasher.update(&signature.to_bytes());
        hasher.update(msg_bytes);
        *items += 1;
        inner.queue(frost_core::batch::Item::new(
            verifying_key.0,
            signature.0,
            msg_bytes,
        )?);
        Ok(self)
    }

    /// Performs batch verification, returning `Ok(_)` if all signatures were
    /// valid and the batch was non-empty, and `Err(_)` otherwise.
    fn verify(self) -> Result<(), SignatureError> {
        let Self {
            hasher,
            inner,
            items,
        } = self;
        let rng = {
            // move hasher so that it can be dropped early automatically
            let mut hasher = hasher;
            hasher.update(&items.to_le_bytes());
            <rand::rngs::ChaCha20Rng as rand::SeedableRng>::from_seed(
                hasher.finalize().into(),
            )
        };
        inner.verify(rng)
    }
}

/// Required for batched verification.
/// It should be safe to re-use the same batch verification context for
/// several batched verifications.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct BatchVerificationContext {
    mac_key: [u8; blake3::KEY_LEN],
}

impl BatchVerificationContext {
    pub fn new<R>(rng: &mut R) -> Self
    where
        R: rand_core::CryptoRng,
    {
        let mut mac_key = [0; blake3::KEY_LEN];
        rng.fill_bytes(&mut mac_key);
        Self { mac_key }
    }

    /// Construct a new batch verifier
    fn verifier(&self) -> BatchVerifier {
        let Self { mac_key } = self;
        BatchVerifier {
            hasher: blake3::Hasher::new_keyed(mac_key),
            inner: frost_core::batch::Verifier::new(),
            items: 0,
        }
    }
}

pub fn get_address(verifying_key: &VerifyingKey) -> Address {
    let mut hasher = blake3::Hasher::new();
    let mut reader = hasher.update(&verifying_key.to_bytes()).finalize_xof();
    let mut output: [u8; 20] = [0; 20];
    reader.fill(&mut output);
    Address(output)
}

/// Canonical message to sign a tx
fn tx_msg_canonical(tx: &Transaction) -> borsh::io::Result<Vec<u8>> {
    let mut buf = vec![Dst::Transaction as u8];
    borsh::to_writer(&mut buf, tx)?;
    Ok(buf)
}

pub fn verify_authorized_transaction(
    ctxt: &BatchVerificationContext,
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    if transaction.authorizations.len() != transaction.transaction.inputs.len()
    {
        return Err(Error::SignatureCountMismatch {
            expected: transaction.transaction.inputs.len(),
            actual: transaction.authorizations.len(),
        });
    }
    let tx_msg_canonical = tx_msg_canonical(&transaction.transaction)?;
    let mut batch_verifier = ctxt.verifier();
    for Authorization {
        verifying_key,
        signature,
    } in &transaction.authorizations
    {
        batch_verifier = batch_verifier.queue_item(
            *verifying_key,
            *signature,
            &tx_msg_canonical,
        )?;
    }
    let () = batch_verifier.verify()?;
    Ok(())
}

pub fn verify_authorizations(
    ctxt: &BatchVerificationContext,
    body: &Body,
) -> Result<(), Error> {
    let input_numbers: Vec<usize> = body
        .transactions
        .iter()
        .map(|transaction| transaction.inputs.len())
        .collect();
    let total_inputs: usize = input_numbers.iter().sum();
    if body.authorizations.len() != total_inputs {
        return Err(Error::SignatureCountMismatch {
            expected: total_inputs,
            actual: body.authorizations.len(),
        });
    }
    if total_inputs == 0 {
        return Ok(());
    }
    let serialized_transactions: Vec<Vec<u8>> = body
        .transactions
        .par_iter()
        .map(tx_msg_canonical)
        .collect::<Result<_, _>>()?;
    let serialized_transactions =
        serialized_transactions.iter().map(Vec::as_slice);
    let messages = input_numbers
        .iter()
        .copied()
        .zip(serialized_transactions)
        .flat_map(|(input_number, serialized_transaction)| {
            std::iter::repeat_n(serialized_transaction, input_number)
        });
    let pairs = body.authorizations.iter().zip(messages).collect::<Vec<_>>();
    const CHUNK_SIZE: usize = 1 << 14;
    pairs.par_chunks(CHUNK_SIZE).try_for_each(|chunk| {
        let mut batch_verifier = ctxt.verifier();
        for (authorization, msg) in chunk {
            let Authorization {
                verifying_key,
                signature,
            } = authorization;
            batch_verifier =
                batch_verifier.queue_item(*verifying_key, *signature, msg)?;
        }
        batch_verifier.verify()
    })?;
    Ok(())
}

/// Sign a message with DST prefix
pub fn sign<R>(
    rng: R,
    signing_key: &SigningKey,
    dst: Dst,
    msg: &[u8],
) -> Signature
where
    R: rand_core::CryptoRng,
{
    let msg_buf = [&[dst as u8], msg].concat();
    Signature(signing_key.sign(rng, &msg_buf))
}

/// Verify a message with DST prefix
pub fn verify(
    signature: Signature,
    verifying_key: &VerifyingKey,
    dst: Dst,
    msg: &[u8],
) -> bool {
    let msg_buf = [&[dst as u8], msg].concat();
    verifying_key.0.verify(&msg_buf, &signature.0).is_ok()
}

pub fn sign_tx<R>(
    rng: R,
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error>
where
    R: rand_core::CryptoRng,
{
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    Ok(sign(
        rng,
        signing_key,
        Dst::Transaction,
        &tx_bytes_canonical,
    ))
}

pub fn authorize<R>(
    mut rng: R,
    addresses_signing_keys: &[(Address, &SigningKey)],
    transaction: Transaction,
) -> Result<AuthorizedTransaction, Error>
where
    R: rand_core::CryptoRng,
{
    let mut authorizations: Vec<Authorization> =
        Vec::with_capacity(addresses_signing_keys.len());
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    for (address, signing_key) in addresses_signing_keys {
        let verifying_key = VerifyingKey::from(*signing_key);
        let hash_verifying_key = get_address(&verifying_key);
        if *address != hash_verifying_key {
            return Err(Error::WrongKeyForAddress {
                address: *address,
                hash_verifying_key,
            });
        }
        let authorization = Authorization {
            verifying_key,
            signature: sign(
                &mut rng,
                signing_key,
                Dst::Transaction,
                &tx_bytes_canonical,
            ),
        };
        authorizations.push(authorization);
    }
    Ok(AuthorizedTransaction {
        authorizations,
        transaction,
        actor_proof: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Authorization, BatchVerificationContext, Dst, Signature, SigningKey,
        authorize, get_address, sign, sign_tx, verify,
        verify_authorized_transaction,
    };
    use crate::types::{
        Address, AuthorizedTransaction, GetAddress as _, Transaction,
        VerifyingKey,
    };
    use const_hex::FromHex as _;

    fn signing_key(seed: u8) -> SigningKey {
        let scalar = curve25519_dalek::Scalar::from_bytes_mod_order([seed; 32]);
        SigningKey::from_scalar(scalar).expect("non-zero scalar")
    }

    fn one_input_tx() -> Transaction {
        Transaction {
            inputs: vec![crate::types::OutPoint::Regular {
                txid: crate::types::Txid::from([3; 32]),
                vout: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn an_address_comes_from_the_compressed_ristretto_point() {
        let key = signing_key(1);
        let verifying_key = VerifyingKey::from(&key);
        let expected = {
            let mut hasher = blake3::Hasher::new();
            let mut reader =
                hasher.update(&verifying_key.to_bytes()).finalize_xof();
            let mut output = [0u8; 20];
            reader.fill(&mut output);
            Address(output)
        };
        assert_eq!(get_address(&verifying_key), expected);
    }

    #[test]
    fn a_verifying_key_round_trips_through_bech32m() {
        let key = VerifyingKey::from(&signing_key(2));
        let encoded = key.bech32m_encode();
        assert_eq!(VerifyingKey::bech32m_decode(&encoded).unwrap(), key);
    }

    #[test]
    fn a_signature_round_trips_through_hex() {
        let signature =
            sign(rand::rng(), &signing_key(3), Dst::Arbitrary, b"hello");
        let encoded = signature.to_string();
        assert_eq!(Signature::from_hex(&encoded).unwrap(), signature);
    }

    #[test]
    fn authorize_puts_the_address_of_the_signer_on_the_authorization() {
        let key = signing_key(4);
        let address = get_address(&VerifyingKey::from(&key));
        let authorized =
            authorize(rand::rng(), &[(address, &key)], one_input_tx())
                .expect("authorize");
        assert_eq!(authorized.authorizations[0].get_address(), address);
    }

    #[test]
    fn authorize_rejects_a_key_that_does_not_match_the_address() {
        let key = signing_key(5);
        let other = get_address(&VerifyingKey::from(&signing_key(6)));
        let err = authorize(rand::rng(), &[(other, &key)], one_input_tx())
            .expect_err("authorize must reject a key for another address");
        assert!(matches!(err, super::Error::WrongKeyForAddress { .. }));
    }

    #[test]
    fn a_good_signature_verifies() {
        let mut rng = rand::rng();
        let key = signing_key(7);
        let address = get_address(&VerifyingKey::from(&key));
        let authorized =
            authorize(&mut rng, &[(address, &key)], one_input_tx())
                .expect("authorize");
        let ctxt = BatchVerificationContext::new(&mut rng);
        verify_authorized_transaction(&ctxt, &authorized)
            .expect("a good signature must verify");
    }

    #[test]
    fn a_signature_from_another_key_fails() {
        let mut rng = rand::rng();
        let victim = signing_key(8);
        let attacker = signing_key(9);
        let transaction = one_input_tx();
        let forged = Authorization {
            verifying_key: VerifyingKey::from(&victim),
            signature: sign_tx(&mut rng, &attacker, &transaction)
                .expect("sign"),
        };
        let authorized = AuthorizedTransaction {
            transaction,
            authorizations: vec![forged],
            actor_proof: None,
        };
        let ctxt = BatchVerificationContext::new(&mut rng);
        assert!(
            verify_authorized_transaction(&ctxt, &authorized).is_err(),
            "a forged signature must not verify"
        );
    }

    #[test]
    fn a_missing_authorization_fails() {
        let mut rng = rand::rng();
        let authorized = AuthorizedTransaction {
            transaction: one_input_tx(),
            authorizations: Vec::new(),
            actor_proof: None,
        };
        let ctxt = BatchVerificationContext::new(&mut rng);
        assert!(matches!(
            verify_authorized_transaction(&ctxt, &authorized),
            Err(super::Error::SignatureCountMismatch { .. })
        ));
    }

    #[test]
    fn a_domain_tag_separates_two_messages() {
        let key = signing_key(10);
        let verifying_key = VerifyingKey::from(&key);
        let signature =
            sign(rand::rng(), &key, Dst::Arbitrary, b"same message");
        assert!(verify(
            signature,
            &verifying_key,
            Dst::Arbitrary,
            b"same message"
        ));
        assert!(
            !verify(
                signature,
                &verifying_key,
                Dst::Transaction,
                b"same message"
            ),
            "a signature under one domain tag must not verify under another"
        );
    }
}
