use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Inputs<Input>(pub Vec<Input>);

impl<Input> Inputs<Input> {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Input> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn push(&mut self, input: Input) {
        self.0.push(input)
    }

    #[inline(always)]
    pub fn remove(&mut self, index: usize) -> Input {
        self.0.remove(index)
    }
}

impl<Input> Default for Inputs<Input> {
    #[inline(always)]
    fn default() -> Self {
        Self(Vec::default())
    }
}

impl<Input> From<Vec<Input>> for Inputs<Input> {
    #[inline(always)]
    fn from(inputs: Vec<Input>) -> Self {
        Self(inputs)
    }
}

impl<'a, Input> IntoIterator for &'a Inputs<Input> {
    type IntoIter = <&'a Vec<Input> as IntoIterator>::IntoIter;
    type Item = &'a Input;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
