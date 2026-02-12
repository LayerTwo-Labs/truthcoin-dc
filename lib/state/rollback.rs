use crate::types::Txid;
use nonempty::NonEmpty;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HeightStamped<T> {
    pub value: T,
    pub height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TxidStamped<T> {
    pub data: T,
    pub txid: Txid,
    pub height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct RollBack<T>(pub(in crate::state) NonEmpty<T>);

impl<T> RollBack<HeightStamped<T>> {
    pub(in crate::state) fn new(value: T, height: u32) -> Self {
        let height_stamped = HeightStamped { value, height };
        Self(NonEmpty::new(height_stamped))
    }

    pub(in crate::state) fn pop(mut self) -> (Option<Self>, HeightStamped<T>) {
        if let Some(value) = self.0.pop() {
            (Some(self), value)
        } else {
            (None, self.0.head)
        }
    }

    pub(in crate::state) fn push(
        &mut self,
        value: T,
        height: u32,
    ) -> Result<(), T> {
        if self.0.last().height > height {
            return Err(value);
        }
        let height_stamped = HeightStamped { value, height };
        self.0.push(height_stamped);
        Ok(())
    }

    pub(in crate::state) fn earliest(&self) -> &HeightStamped<T> {
        self.0.first()
    }

    pub fn latest(&self) -> &HeightStamped<T> {
        self.0.last()
    }
}

impl<T> RollBack<TxidStamped<T>> {
    pub(in crate::state) fn new(value: T, txid: Txid, height: u32) -> Self {
        let txid_stamped = TxidStamped {
            data: value,
            txid,
            height,
        };
        Self(NonEmpty::new(txid_stamped))
    }

    pub(in crate::state) fn pop(&mut self) -> Option<TxidStamped<T>> {
        self.0.pop()
    }

    pub(in crate::state) fn push(&mut self, value: T, txid: Txid, height: u32) {
        let txid_stamped = TxidStamped {
            data: value,
            txid,
            height,
        };
        self.0.push(txid_stamped)
    }

    pub(in crate::state) fn at_block_height(
        &self,
        height: u32,
    ) -> Option<&TxidStamped<T>> {
        self.0
            .iter()
            .rev()
            .find(|txid_stamped| txid_stamped.height <= height)
    }

    pub fn latest(&self) -> &TxidStamped<T> {
        self.0.last()
    }
}
