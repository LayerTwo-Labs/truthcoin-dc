use thiserror::Error;

#[derive(Debug, Error)]
#[error("invalid decision ID: {reason}")]
pub struct InvalidDecisionId {
    pub reason: String,
}
