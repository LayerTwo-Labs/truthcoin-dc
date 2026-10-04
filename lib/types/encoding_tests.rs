//! Fixed encodings of the types that peers send and that the databases
//! hold. A change to one of these bytes makes old datadirs unreadable, or
//! splits the network.

use bitcoin::hashes::Hash as _;

use crate::{
    state::{
        decisions::{DecisionId, DecisionType},
        markets::{DimensionSpec, MarketId},
    },
    types::{
        Address, Block, BlockHash, Body, ClaimDecisionPayload, Coinbase,
        CoinbaseTxid, DecisionClaimEntry, Header, InPoint, M6id, MerkleRoot,
        OutPoint, OutPointKey, Output, OutputContent, SpentOutput, Transaction,
        TransactionData, Txid,
    },
};

fn outpoints() -> [OutPoint; 5] {
    [
        OutPoint::Regular {
            txid: Txid([1; 32]),
            vout: 2,
        },
        OutPoint::Coinbase {
            txid: CoinbaseTxid([3; 32]),
            vout: 4,
        },
        OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([5; 32]),
            vout: 6,
        }),
        OutPoint::MarketFunds {
            market_id: [7; 6],
            block_height: 8,
            is_fee: true,
        },
        OutPoint::Payout {
            hash: MerkleRoot::from([9; 32]),
            vout: 10,
        },
    ]
}

fn withdrawal_address()
-> anyhow::Result<bitcoin::Address<bitcoin::address::NetworkUnchecked>> {
    Ok("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq".parse()?)
}

fn outputs() -> anyhow::Result<[Output; 3]> {
    Ok([
        Output {
            address: Address([11; 20]),
            content: OutputContent::Value(bitcoin::Amount::from_sat(12)),
        },
        Output {
            address: Address([13; 20]),
            content: OutputContent::Withdrawal {
                value: bitcoin::Amount::from_sat(14),
                main_fee: bitcoin::Amount::from_sat(15),
                main_address: withdrawal_address()?,
            },
        },
        Output {
            address: Address([18; 20]),
            content: OutputContent::MarketFunds {
                market_id: [19; 6],
                amount: bitcoin::Amount::from_sat(20),
                is_fee: false,
            },
        },
    ])
}

fn transactions() -> anyhow::Result<[Transaction; 2]> {
    let [regular, coinbase, deposit, market_funds, payout] = outpoints();
    let trade = Transaction {
        inputs: vec![
            (regular, [44; 32]),
            (coinbase, [45; 32]),
            (deposit, [46; 32]),
        ]
        .into(),
        proof: Default::default(),
        outputs: outputs()?.to_vec().into(),
        data: Some(TransactionData::Trade {
            market_id: MarketId::new([22; 6]),
            outcome_index: 23,
            shares: -24,
            trader: Address([25; 20]),
            limit_sats: 26,
            tx_pow_nonce: Some(27),
            prev_block_hash: BlockHash([28; 32]),
        }),
    };
    let create_market = Transaction {
        inputs: vec![(market_funds, [47; 32]), (payout, [48; 32])].into(),
        proof: Default::default(),
        outputs: Vec::new().into(),
        data: Some(TransactionData::CreateMarket {
            title: "title".to_owned(),
            description: "description".to_owned(),
            dimension_specs: vec![
                DimensionSpec::Single(DecisionId::new(true, 1, 0)?),
                DimensionSpec::Categorical(DecisionId::new(false, 2, 3)?),
            ],
            new_claims: vec![ClaimDecisionPayload {
                decision_type: DecisionType::Scaled {
                    min: 0.0,
                    max: 100.0,
                    increment: 0.5,
                },
                decisions: vec![DecisionClaimEntry {
                    decision_id_bytes: [0x81, 0, 0],
                    header: "header".to_owned(),
                    description: "description".to_owned(),
                    option_0_label: Some("no".to_owned()),
                    option_1_label: None,
                    option_labels: Some(vec!["a".to_owned(), "b".to_owned()]),
                    tags: None,
                }],
            }],
            trading_fee: Some(0.01),
            tx_pow_hash_selector: Some(29),
            tx_pow_ordering: None,
            tx_pow_difficulty: Some(30),
        }),
    };
    Ok([trade, create_market])
}

fn block() -> anyhow::Result<Block> {
    let header = Header {
        merkle_root: MerkleRoot::from([31; 32]),
        prev_side_hash: Some(BlockHash([32; 32])),
        prev_main_hash: bitcoin::BlockHash::from_byte_array([33; 32]),
        roots: Vec::new(),
    };
    let body = Body {
        coinbase: Coinbase {
            memo: vec![16, 17],
            outputs: outputs()?.to_vec().into(),
        },
        transactions: transactions()?.to_vec(),
        authorizations: Vec::new(),
        actor_proofs: vec![None, None],
    };
    Ok(Block {
        header,
        body,
        height: 34,
    })
}

fn filled_outputs() -> anyhow::Result<[Output; 3]> {
    Ok([
        Output {
            address: Address([35; 20]),
            content: OutputContent::Value(bitcoin::Amount::from_sat(36)),
        },
        Output {
            address: Address([37; 20]),
            content: OutputContent::Withdrawal {
                value: bitcoin::Amount::from_sat(38),
                main_fee: bitcoin::Amount::from_sat(39),
                main_address: withdrawal_address()?,
            },
        },
        Output {
            address: Address([40; 20]),
            content: OutputContent::MarketFunds {
                market_id: [41; 6],
                amount: bitcoin::Amount::from_sat(42),
                is_fee: true,
            },
        },
    ])
}

/// Borsh, bincode and database key encodings of each outpoint, as hex
#[test]
fn outpoint_encodings() -> anyhow::Result<()> {
    const EXPECTED: [&str; 15] =
        ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""];
    let mut encodings = Vec::new();
    for outpoint in outpoints() {
        encodings.push(const_hex::encode(borsh::to_vec(&outpoint)?));
        encodings.push(const_hex::encode(bincode::serialize(&outpoint)?));
        encodings.push(const_hex::encode(OutPointKey::from(&outpoint)));
    }
    assert_eq!(encodings, EXPECTED);
    Ok(())
}

/// Txid, borsh and bincode encodings of each transaction, as hex
#[test]
fn transaction_encodings() -> anyhow::Result<()> {
    const EXPECTED: [&str; 6] = ["", "", "", "", "", ""];
    let mut encodings = Vec::new();
    for tx in transactions()? {
        encodings.push(tx.txid().to_string());
        encodings.push(const_hex::encode(borsh::to_vec(&tx)?));
        encodings.push(const_hex::encode(bincode::serialize(&tx)?));
    }
    assert_eq!(encodings, EXPECTED);
    Ok(())
}

/// Header hash, header and body encodings, and block bincode and JSON
#[test]
fn block_encodings() -> anyhow::Result<()> {
    const EXPECTED: [&str; 7] = ["", "", "", "", "", "", ""];
    let block = block()?;
    let encodings = [
        block.header.hash().to_string(),
        const_hex::encode(borsh::to_vec(&block.header)?),
        const_hex::encode(bincode::serialize(&block.header)?),
        const_hex::encode(borsh::to_vec(&block.body)?),
        const_hex::encode(bincode::serialize(&block.body)?),
        const_hex::encode(bincode::serialize(&block)?),
        serde_json::to_string(&block)?,
    ];
    assert_eq!(encodings, EXPECTED);
    Ok(())
}

/// Bincode encoding of each spent output, as the stxo database holds it
#[test]
fn spent_output_encodings() -> anyhow::Result<()> {
    const EXPECTED: [&str; 3] = ["", "", ""];
    let mut encodings = Vec::new();
    for output in filled_outputs()? {
        let spent_output = SpentOutput {
            output,
            inpoint: InPoint::Withdrawal {
                m6id: M6id(bitcoin::Txid::from_byte_array([43; 32])),
            },
        };
        encodings.push(const_hex::encode(bincode::serialize(&spent_output)?));
    }
    assert_eq!(encodings, EXPECTED);
    Ok(())
}
