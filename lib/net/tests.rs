use std::{
    collections::HashSet,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use anyhow::Context as _;
use futures::StreamExt as _;
use heed::types::{SerdeBincode, Unit};

use super::{
    ALPHANET_SEED_NODE_ADDRS, Archive, DatabaseUnique, DialSeedsHandle, Net,
    Network, PeerConnectionInfo, PeerInfoRx, State,
};
use crate::types::net::{ResolvedSeedAddress, SeedAddress};

pub(crate) fn set_crypto_provider() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("install the test TLS provider");
    });
}

#[test]
fn alphanet_names_the_seed_port() {
    assert_eq!(
        ALPHANET_SEED_NODE_ADDRS,
        [SeedAddress {
            host: url::Host::Domain("seed.alpha.ecash.eu.com"),
            port: 4013,
        }]
    );
}

#[tokio::test]
async fn seed_resolution_keeps_the_port() -> anyhow::Result<()> {
    let dns_resolver = hickory_resolver::Resolver::builder_tokio()?.build()?;
    let seed_addr: SeedAddress = "localhost:4013".parse()?;
    let resolved =
        super::resolve_seed_address(&dns_resolver, seed_addr).await?;
    assert_eq!(resolved.port(), 4013);
    assert!(resolved.ip_addrs().all(|addr| addr.is_loopback()));
    Ok(())
}

#[test]
fn seed_resolution_keeps_both_address_families() {
    let v4 = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let v6 = IpAddr::V6(Ipv6Addr::LOCALHOST);
    let resolved = ResolvedSeedAddress::Domain {
        domain: "localhost".to_owned(),
        port: 4013,
        addrs: nonempty::NonEmpty {
            head: v4,
            tail: vec![v6],
        },
    };
    assert_eq!(resolved.ip_addrs().collect::<Vec<_>>(), [v6, v4]);
    let (first, rest) = resolved.pop_first_ip_addr();
    assert_eq!(first, v6);
    assert_eq!(rest.map(|rest| rest.first_ip_addr()), Some(v4));
}

/// Every seed reaches a peer table that already exists, and a second call
/// writes the same set.
#[test]
fn seeds_reach_an_existing_database() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
    options.map_size(16 * 1024 * 1024).max_dbs(2);
    let env = unsafe { sneed::Env::open(&options, dir.path()) }?;
    let network = Network::Signet;
    let known_peers = {
        let mut rwtxn = env.write_txn()?;
        let known_peers: DatabaseUnique<SerdeBincode<SocketAddr>, Unit> =
            DatabaseUnique::create(&env, &mut rwtxn, "known_peers")?;
        super::ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
        super::ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
        rwtxn.commit()?;
        known_peers
    };
    let rotxn = env.read_txn()?;
    let seed_socket_addrs: Vec<SocketAddr> = super::seed_node_addrs(network)
        .iter()
        .filter_map(SeedAddress::socket_addr)
        .collect();
    for seed_node_addr in &seed_socket_addrs {
        anyhow::ensure!(
            known_peers.try_get(&rotxn, seed_node_addr)?.is_some(),
            "the seed {seed_node_addr} never reached the database"
        );
    }
    assert_eq!(known_peers.len(&rotxn)?, seed_socket_addrs.len() as u64);
    Ok(())
}

#[test]
fn ipv4_node_connects_with_both_seed_families() {
    set_crypto_provider();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
        options
            .map_size(16 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS + Net::NUM_DBS);
        let env = unsafe { sneed::Env::open(&options, dir.path()) }.unwrap();
        let state = State::new(&env, None).unwrap();
        let archive = Archive::new(&env).unwrap();
        let (remote, _) = super::make_server_endpoint(
            "127.0.0.1:0".parse().unwrap(),
            HashSet::new(),
        )
        .unwrap();
        let ipv4 = remote.local_addr().unwrap();
        let ipv6 = SocketAddr::new("::1".parse().unwrap(), ipv4.port());
        {
            let mut txn = env.write_txn().unwrap();
            let peers = DatabaseUnique::<
                heed::types::SerdeBincode<SocketAddr>,
                heed::types::Unit,
            >::create(&env, &mut txn, "known_peers")
            .unwrap();
            for addr in [ipv6, ipv4] {
                peers.put(&mut txn, &addr, &()).unwrap();
            }
            txn.commit().unwrap();
        }
        let (net, _events, _dial_seeds) = Net::new(
            &tokio::runtime::Handle::current(),
            &env,
            archive,
            None,
            Network::Regtest,
            state,
            "0.0.0.0:0".parse().unwrap(),
            HashSet::new(),
            HashSet::new(),
        )
        .unwrap();
        let connection = tokio::time::timeout(Duration::from_secs(3), async {
            remote.accept().await.unwrap().await.unwrap()
        })
        .await
        .unwrap();
        assert!(connection.remote_address().is_ipv4());
        let peers = net.get_active_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].address, ipv4);
        net.server.close(0_u32.into(), b"test complete");
        remote.close(0_u32.into(), b"test complete");
    });
}

#[tokio::test]
async fn rejected_duplicate_has_no_peer_close_event() -> anyhow::Result<()> {
    use futures::StreamExt;

    set_crypto_provider();
    let dir = tempfile::tempdir()?;
    let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
    options
        .map_size(16 * 1024 * 1024)
        .max_dbs(State::NUM_DBS + Archive::NUM_DBS + Net::NUM_DBS);
    let env = unsafe { sneed::Env::open(&options, dir.path()) }?;
    let state = State::new(&env, None)?;
    let archive = Archive::new(&env)?;
    let (net, info_rx, _dial_seeds) = Net::new(
        &tokio::runtime::Handle::current(),
        &env,
        archive,
        None,
        Network::Regtest,
        state,
        "127.0.0.1:0".parse()?,
        HashSet::new(),
        HashSet::new(),
    )?;
    let (remote, _) =
        super::make_server_endpoint("127.0.0.1:0".parse()?, HashSet::new())?;
    let addr = remote.local_addr()?;
    net.connect_peer(env.clone(), addr.into())?;
    let context = super::PeerConnectionCtxt {
        env,
        archive: net.archive.clone(),
        magic_bytes: net.magic_bytes,
        resolved_address: addr.into(),
        state: net.state.clone(),
    };
    let (duplicate, duplicate_info) =
        super::peer::connect(net.server.connect(addr, "localhost")?, context);

    let error = net
        .add_active_peer(addr, duplicate, duplicate_info)
        .unwrap_err();
    assert_eq!(error.0, addr);
    assert_eq!(net.get_active_peers().len(), 1);
    drop(net);

    let events = tokio::time::timeout(
        Duration::from_secs(5),
        info_rx.collect::<Vec<_>>(),
    )
    .await?;
    assert_eq!(events.iter().filter(|(_, info)| info.is_none()).count(), 1);
    Ok(())
}

fn temp_net() -> anyhow::Result<(
    tempfile::TempDir,
    sneed::Env<heed::WithoutTls>,
    Net,
    PeerInfoRx,
)> {
    let (dir, env, net, info_rx, _dial_seeds) =
        temp_net_with_peers(HashSet::new())?;
    Ok((dir, env, net, info_rx))
}

fn temp_net_with_peers(
    add_peers: HashSet<SeedAddress>,
) -> anyhow::Result<(
    tempfile::TempDir,
    sneed::Env<heed::WithoutTls>,
    Net,
    PeerInfoRx,
    DialSeedsHandle,
)> {
    set_crypto_provider();
    let dir = tempfile::tempdir()?;
    let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
    options
        .map_size(16 * 1024 * 1024)
        .max_dbs(State::NUM_DBS + Archive::NUM_DBS + Net::NUM_DBS);
    let env = unsafe { sneed::Env::open(&options, dir.path()) }?;
    let state = State::new(&env, None)?;
    let archive = Archive::new(&env)?;
    let (net, info_rx, dial_seeds) = Net::new(
        &tokio::runtime::Handle::current(),
        &env,
        archive,
        None,
        Network::Regtest,
        state,
        "127.0.0.1:0".parse()?,
        add_peers,
        HashSet::new(),
    )?;
    Ok((dir, env, net, info_rx, dial_seeds))
}

#[tokio::test]
async fn connect_peer_keeps_a_static_ipv4_address() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::new(),
    )?;
    let addr = remote.local_addr()?;

    net.connect_peer(env, addr.into())?;

    let peers = net.get_active_peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].address, addr);
    Ok(())
}

#[tokio::test]
async fn connect_peer_returns_other_quinn_errors() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    net.server.close(0_u32.into(), b"test complete");
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 4004));

    let error = net.connect_peer(env, addr.into()).unwrap_err();

    assert!(matches!(
        error,
        super::Error::Connect(quinn::ConnectError::EndpointStopping)
    ));
    assert!(net.get_active_peers().is_empty());
    Ok(())
}

#[tokio::test]
async fn connect_peer_skips_ipv6_on_an_ipv4_endpoint() -> anyhow::Result<()> {
    let (_dir, env, net, mut info_rx) = temp_net()?;
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::new(),
    )?;
    let addr = remote.local_addr()?;
    let next_ip = Ipv4Addr::new(127, 0, 0, 2);
    let resolved = ResolvedSeedAddress::Domain {
        domain: "localhost".to_owned(),
        port: addr.port(),
        addrs: nonempty::NonEmpty {
            head: next_ip.into(),
            tail: vec![Ipv4Addr::LOCALHOST.into(), Ipv6Addr::LOCALHOST.into()],
        },
    };

    net.connect_peer(env, resolved)?;

    let peers = net.get_active_peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].address, addr);
    assert!(net.server.local_addr()?.is_ipv4());
    net.server.close(0_u32.into(), b"test complete");
    let (reported_addr, info) =
        tokio::time::timeout(Duration::from_secs(5), info_rx.next())
            .await?
            .context("the peer task returned no result")?;
    let Some(PeerConnectionInfo::Error { resolved_addr, .. }) = info else {
        anyhow::bail!("the peer task returned no connection error");
    };
    assert_eq!(reported_addr, addr);
    assert_eq!(
        resolved_addr.ip_addrs().collect::<Vec<_>>(),
        vec![IpAddr::V4(Ipv4Addr::LOCALHOST), next_ip.into()]
    );
    Ok(())
}

#[tokio::test]
async fn connect_peer_returns_the_last_invalid_address() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    let addr = SocketAddr::from((Ipv6Addr::LOCALHOST, 4004));
    let resolved = ResolvedSeedAddress::Domain {
        domain: "localhost".to_owned(),
        port: addr.port(),
        addrs: nonempty::NonEmpty {
            head: addr.ip(),
            tail: vec!["::2".parse()?],
        },
    };

    let error = net.connect_peer(env, resolved).unwrap_err();

    assert!(matches!(
        error,
        super::Error::Connect(
            quinn::ConnectError::InvalidRemoteAddress(failed)
        ) if failed == addr
    ));
    assert!(net.get_active_peers().is_empty());
    Ok(())
}

/// A seed host name resolves at startup, the node dials it, and the database
/// holds no resolved address for it.
#[tokio::test]
async fn a_seed_host_name_dials_at_startup() -> anyhow::Result<()> {
    set_crypto_provider();
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::new(),
    )?;
    let addr = remote.local_addr()?;
    let seed_addr: SeedAddress =
        format!("localhost:{}", addr.port()).parse()?;
    let (_dir, env, net, _info_rx, _dial_seeds) =
        temp_net_with_peers(HashSet::from([seed_addr]))?;

    tokio::time::timeout(Duration::from_secs(5), async {
        while net.get_active_peers().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("the node did not dial the seed host name")?;

    let peers = net.get_active_peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].address, addr);
    let rotxn = env.read_txn()?;
    assert_eq!(net.known_peers.len(&rotxn)?, 0);
    Ok(())
}

const TEST_REDIAL_MIN_DELAY: Duration = Duration::from_millis(50);
const TEST_REDIAL_MAX_DELAY: Duration = Duration::from_millis(200);

fn resolver_with_hosts(
    hosts_conf: &str,
) -> anyhow::Result<hickory_resolver::TokioResolver> {
    let mut hosts = hickory_resolver::Hosts::default();
    hosts.read_hosts_conf(hosts_conf.as_bytes())?;
    let mut resolver = hickory_resolver::Resolver::builder_tokio()?.build()?;
    resolver.set_hosts(Arc::new(hosts));
    Ok(resolver)
}

async fn wait_for_a_peer(net: &Net) -> anyhow::Result<()> {
    tokio::time::timeout(Duration::from_secs(5), async {
        while net.get_active_peers().is_empty() {
            tokio::time::sleep(TEST_REDIAL_MIN_DELAY).await;
        }
    })
    .await
    .context("the node dialed no peer again")?;
    Ok(())
}

/// A peer that drops leaves no connection, so the node dials it again.
#[tokio::test]
async fn a_lost_peer_is_dialed_again() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::new(),
    )?;
    let addr = remote.local_addr()?;
    net.connect_peer(env.clone(), addr.into())?;
    assert_eq!(net.get_active_peers().len(), 1);
    net.remove_active_peer(addr);
    assert!(net.get_active_peers().is_empty());

    let redial = tokio::spawn({
        let net = net.clone();
        async move {
            net.redial_known_peers(
                env,
                TEST_REDIAL_MIN_DELAY,
                TEST_REDIAL_MAX_DELAY,
            )
            .await
        }
    });
    let dialed_again = wait_for_a_peer(&net).await;
    redial.abort();

    dialed_again?;
    assert_eq!(net.get_active_peers()[0].address, addr);
    Ok(())
}

/// A peer that holds a connection takes no redial, and the loop starts no
/// second connection to it.
#[tokio::test]
async fn a_connected_peer_takes_no_redial() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::new(),
    )?;
    let addr = remote.local_addr()?;
    net.connect_peer(env.clone(), addr.into())?;

    assert!(!net.dial_peer(env.clone(), addr.into())?);
    assert_eq!(net.dial_known_peers(&env).await?, 0);

    let redial = tokio::spawn({
        let net = net.clone();
        let env = env.clone();
        async move {
            net.redial_known_peers(
                env,
                TEST_REDIAL_MIN_DELAY,
                TEST_REDIAL_MAX_DELAY,
            )
            .await
        }
    });
    tokio::time::sleep(TEST_REDIAL_MAX_DELAY * 5).await;
    redial.abort();

    let peers = net.get_active_peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].address, addr);
    Ok(())
}

/// A seed host name resolves again at each redial, so the node reaches a seed
/// that moved to another address.
#[tokio::test]
async fn a_seed_host_name_dials_its_new_address() -> anyhow::Result<()> {
    let domain = "seed.truthcoin.test";
    let (_dir, env, mut net, _info_rx) = temp_net()?;
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::from([domain.to_owned()]),
    )?;
    let addr = remote.local_addr()?;
    let seed_addr: SeedAddress = format!("{domain}:{}", addr.port()).parse()?;
    net.seed_names = Arc::new(HashSet::from([seed_addr.clone()]));

    net.dns_resolver =
        Arc::new(resolver_with_hosts(&format!("0.0.0.0 {domain}"))?);
    let error = net.dial_seed(env.clone(), seed_addr).await.unwrap_err();
    assert!(matches!(error, super::error::DialSeed::DnsResolve(_)));
    assert!(net.get_active_peers().is_empty());

    net.dns_resolver =
        Arc::new(resolver_with_hosts(&format!("127.0.0.1 {domain}"))?);
    let redial = tokio::spawn({
        let net = net.clone();
        let env = env.clone();
        async move {
            net.redial_known_peers(
                env,
                TEST_REDIAL_MIN_DELAY,
                TEST_REDIAL_MAX_DELAY,
            )
            .await
        }
    });
    let dialed_again = wait_for_a_peer(&net).await;
    redial.abort();

    dialed_again?;
    assert_eq!(net.get_active_peers()[0].address, addr);
    let rotxn = env.read_txn()?;
    assert_eq!(net.known_peers.len(&rotxn)?, 0);
    Ok(())
}

/// The QUIC server name of a host name peer is the domain.
#[tokio::test]
async fn connect_peer_sends_the_domain_as_server_name() -> anyhow::Result<()> {
    let (_dir, env, net, _info_rx) = temp_net()?;
    let domain = "seed.truthcoin.test";
    let (remote, _) = super::make_server_endpoint(
        (Ipv4Addr::LOCALHOST, 0).into(),
        HashSet::from([domain.to_owned()]),
    )?;
    let addr = remote.local_addr()?;
    let resolved = ResolvedSeedAddress::Domain {
        domain: domain.to_owned(),
        port: addr.port(),
        addrs: nonempty::NonEmpty::new(addr.ip()),
    };

    net.connect_peer(env, resolved)?;

    let connection =
        tokio::time::timeout(Duration::from_secs(5), remote.accept())
            .await?
            .context("the endpoint closed before the connection")?
            .await?;
    let handshake_data = connection
        .handshake_data()
        .context("the connection holds no handshake data")?
        .downcast::<quinn::crypto::rustls::HandshakeData>()
        .map_err(|_| anyhow::anyhow!("the handshake data is not rustls"))?;
    assert_eq!(handshake_data.server_name.as_deref(), Some(domain));
    remote.close(0_u32.into(), b"test complete");
    Ok(())
}
