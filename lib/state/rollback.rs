use nonempty::NonEmpty;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HeightStamped<T> {
    pub value: T,
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
