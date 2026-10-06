//! Mostly a wrapper around truthcoin_dc_types

use std::sync::LazyLock;

pub use truthcoin_dc_types::*;

#[cfg(test)]
mod encoding_tests;
pub mod proto;

// Do not make this public outside of this crate, as it could break semver
pub(crate) static VERSION: LazyLock<Version> = LazyLock::new(|| {
    const VERSION_STR: &str = env!("CARGO_PKG_VERSION");
    semver::Version::parse(VERSION_STR).unwrap().into()
});
