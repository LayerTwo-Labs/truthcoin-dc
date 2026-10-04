//! UTXO writes that record their Utreexo leaf in an accumulator diff

use fallible_iterator::FallibleIterator as _;
use sneed::RwTxn;

use crate::{
    state::{Error, State},
    types::{AccumulatorDiff, OutPoint, OutPointKey, Output, PointedOutputRef},
};

/// Writes to the UTXO set. Each write records its Utreexo leaf in the
/// accumulator diff.
pub trait UtxoManager {
    fn insert_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        output: &Output,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<(), Error>;
    fn delete_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<bool, Error>;
    fn clear_utxos(
        &self,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<(), Error>;
}

impl UtxoManager for State {
    fn insert_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        output: &Output,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<(), Error> {
        let key = OutPointKey::from(outpoint);
        self.utxos.put(rwtxn, &key, output)?;
        accumulator_diff.insert(
            PointedOutputRef {
                outpoint: *outpoint,
                output,
            }
            .into(),
        );
        Ok(())
    }

    fn delete_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<bool, Error> {
        let key = OutPointKey::from(outpoint);
        let output = if let Some(output) = self.utxos.try_get(rwtxn, &key)? {
            output
        } else {
            return Ok(false);
        };
        if !self.utxos.delete(rwtxn, &key)? {
            return Ok(false);
        }
        accumulator_diff.remove(
            PointedOutputRef {
                outpoint: *outpoint,
                output: &output,
            }
            .into(),
        );
        Ok(true)
    }

    fn clear_utxos(
        &self,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
    ) -> Result<(), Error> {
        let mut iter = self.utxos.iter(rwtxn)?;
        while let Some((key, output)) = iter.next()? {
            let outpoint = OutPoint::from(key);
            accumulator_diff.remove(
                PointedOutputRef {
                    outpoint,
                    output: &output,
                }
                .into(),
            );
        }
        drop(iter);
        self.utxos.clear(rwtxn)?;
        Ok(())
    }
}
