use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    address::Address,
    authorization::{self, Authorization, Dst},
    hashes::{self, OfferId},
    transaction::GetAddress,
    votecoin::Votecoin,
};

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct VotecoinOffer {
    pub seller: Address,
    pub amount: Votecoin,
    #[borsh(serialize_with = "crate::util::borsh::serialize::bitcoin_amount")]
    #[serde(rename = "price_sats", with = "bitcoin::amount::serde::as_sat")]
    #[schema(value_type = u64)]
    pub price: bitcoin::Amount,
    pub pay_to: Address,
    pub voting_period: u32,
    pub nonce: u64,
}

impl VotecoinOffer {
    pub fn id(&self) -> OfferId {
        hashes::hash(self).into()
    }

    pub fn signing_message(&self) -> borsh::io::Result<Vec<u8>> {
        borsh::to_vec(self)
    }
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, PartialEq, Serialize, ToSchema,
)]
pub struct SignedOffer {
    pub offer: VotecoinOffer,
    pub authorization: Authorization,
}

impl SignedOffer {
    pub fn id(&self) -> OfferId {
        self.offer.id()
    }

    pub fn is_signed_by_seller(&self) -> bool {
        let Ok(msg) = self.offer.signing_message() else {
            return false;
        };
        self.authorization.get_address() == self.offer.seller
            && authorization::verify(
                self.authorization.signature,
                &self.authorization.verifying_key,
                Dst::VotecoinOffer,
                &msg,
            )
    }
}
