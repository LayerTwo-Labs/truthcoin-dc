//! UTXO writes that record their Utreexo leaf in an accumulator diff

use fallible_iterator::FallibleIterator as _;
use sneed::RwTxn;

use crate::{
    state::{Error, State},
    types::{
        AccumulatorDiff, OutPoint, OutPointKey, Output, PointedOutputRef,
        state::{MarketUtxo, MarketUtxoChanges, MarketUtxoReason},
    },
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

impl State {
    /// Insert an output that no transaction creates, and record it for the
    /// block index
    pub(in crate::state) fn insert_market_utxo(
        &self,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
        market_utxos: &mut MarketUtxoChanges,
        utxo: MarketUtxo,
    ) -> Result<(), Error> {
        self.insert_utxo(
            rwtxn,
            &utxo.outpoint,
            &utxo.output,
            accumulator_diff,
        )?;
        market_utxos.creates.push(utxo);
        Ok(())
    }

    /// Delete an output that no transaction spends, and record it for the
    /// block index. Returns false if the UTXO set does not hold it.
    pub(in crate::state) fn delete_market_utxo(
        &self,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
        market_utxos: &mut MarketUtxoChanges,
        outpoint: &OutPoint,
        reason: MarketUtxoReason,
    ) -> Result<bool, Error> {
        let Some(output) =
            self.utxos.try_get(rwtxn, &OutPointKey::from(outpoint))?
        else {
            return Ok(false);
        };
        if !self.delete_utxo(rwtxn, outpoint, accumulator_diff)? {
            return Ok(false);
        }
        market_utxos.deletes.push(MarketUtxo {
            outpoint: *outpoint,
            output,
            reason,
        });
        Ok(true)
    }
}
