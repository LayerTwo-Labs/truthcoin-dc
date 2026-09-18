use std::{
    collections::{HashMap, HashSet, hash_map},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use fallible_iterator::FallibleIterator;
use futures::{StreamExt, channel::mpsc};
use heed::types::{SerdeBincode, Unit};
use hickory_resolver::TokioResolver;
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use sneed::{
    DatabaseUnique, DbError, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey,
};
use tokio_stream::StreamNotifyClose;
use tracing::instrument;

use crate::{
    archive::Archive,
    state::State,
    types::{
        AuthorizedTransaction, Network, VERSION, Version,
        net::{DEFAULT_PORT, ResolvedSeedAddress, SeedAddress},
    },
    util::ErrorChain,
};

pub mod error;
mod peer;

pub use error::Error;
pub(crate) use peer::error::mailbox::Error as PeerConnectionMailboxError;
use peer::{
    Connection, ConnectionContext as PeerConnectionCtxt,
    ConnectionHandle as PeerConnectionHandle,
};
pub use peer::{
    ConnectionError as PeerConnectionError, Info as PeerConnectionInfo,
    InternalMessage as PeerConnectionMessage, Peer, PeerConnectionStatus,
    PeerStateId, Request as PeerRequest, ResponseMessage as PeerResponse,
    message as peer_message,
};

#[cfg(test)]
pub(crate) mod tests;

/// Dummy certificate verifier that treats any certificate as valid.
/// NOTE, such verification is vulnerable to MITM attacks, but convenient for testing.
#[derive(Debug)]
struct SkipServerVerification;
impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}
impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer,
        _intermediates: &[rustls::pki_types::CertificateDer],
        _server_name: &rustls::pki_types::ServerName,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn configure_client() -> Result<ClientConfig, error::ConfigureClient> {
    let crypto_provider = Arc::new(rustls::crypto::ring::default_provider());
    let crypto = rustls::ClientConfig::builder_with_provider(crypto_provider)
        .with_safe_default_protocol_versions()
        .map_err(error::configure_client::Inner::Rustls)?
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    let client_config =
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
    Ok(ClientConfig::new(Arc::new(client_config)))
}
/// Returns default server configuration along with its certificate.
fn configure_server(
    mut server_names: HashSet<String>,
) -> Result<(ServerConfig, Vec<u8>), Error> {
    server_names.insert("localhost".to_owned());
    let server_names = Vec::from_iter(server_names);
    let cert_key = rcgen::generate_simple_self_signed(server_names)?;
    let keypair_der = cert_key.key_pair.serialize_der();
    let priv_key = rustls::pki_types::PrivateKeyDer::Pkcs8(keypair_der.into());
    let cert_der = cert_key.cert.der().to_vec();
    let cert_chain = vec![cert_key.cert.into()];
    let mut server_config =
        ServerConfig::with_single_cert(cert_chain, priv_key)?;
    let transport_config = Arc::get_mut(&mut server_config.transport)
        .ok_or(Error::TransportConfigNotExclusive)?;
    transport_config.max_concurrent_uni_streams(1_u8.into());
    Ok((server_config, cert_der))
}
/// Constructs a QUIC endpoint configured to listen for incoming connections on a certain address
/// and port.
///
/// ## Returns
///
/// - a stream of incoming QUIC connections
/// - server certificate serialized into DER format
pub fn make_server_endpoint(
    bind_addr: SocketAddr,
    server_names: HashSet<String>,
) -> Result<(Endpoint, Vec<u8>), Error> {
    let (server_config, server_cert) = configure_server(server_names)?;
    tracing::info!(%bind_addr, "creating server endpoint");
    let mut endpoint = Endpoint::server(server_config, bind_addr)?;
    let client_cfg = configure_client()?;
    endpoint.set_default_client_config(client_cfg);
    Ok((endpoint, server_cert))
}

// None indicates that the stream has ended
pub type PeerInfoRx =
    mpsc::UnboundedReceiver<(SocketAddr, Option<PeerConnectionInfo>)>;

const SIGNET_SEED_NODE_ADDRS: &[SeedAddress<&str>] = {
    const SIGNET_MINING_SERVER: SeedAddress<&str> = SeedAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(172, 105, 148, 135)),
        port: DEFAULT_PORT,
    };
    // truthcoin.bip300.xyz
    const BIP300_XYZ: SeedAddress<&str> = SeedAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(95, 217, 243, 12)),
        port: DEFAULT_PORT,
    };
    &[SIGNET_MINING_SERVER, BIP300_XYZ]
};

const ALPHANET_SEED_NODE_ADDRS: &[SeedAddress<&str>] = {
    const ALPHANET_SEED: SeedAddress<&str> = SeedAddress {
        host: url::Host::Domain("seed.alpha.ecash.eu.com"),
        port: DEFAULT_PORT,
    };
    &[ALPHANET_SEED]
};

const FORKNET_SEED_NODE_ADDRS: &[SeedAddress<&str>] = {
    // explorer.bip300.xyz
    const BIP300_XYZ: SeedAddress<&str> = SeedAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(157, 180, 8, 224)),
        port: DEFAULT_PORT,
    };
    &[BIP300_XYZ]
};

const fn seed_node_addrs(
    network: Network,
) -> &'static [SeedAddress<&'static str>] {
    match network {
        Network::Signet => SIGNET_SEED_NODE_ADDRS,
        Network::Regtest => &[],
        Network::Alphanet => ALPHANET_SEED_NODE_ADDRS,
        Network::Forknet => FORKNET_SEED_NODE_ADDRS,
    }
}

/// Add every seed IP address the network names that the database does not
/// hold. A datadir made before a seed existed would otherwise never learn it.
/// A seed host name stays out of the database, and resolves at each start.
fn ensure_seed_peers(
    known_peers: &DatabaseUnique<SerdeBincode<SocketAddr>, Unit>,
    rwtxn: &mut RwTxn,
    network: Network,
) -> Result<(), DbError> {
    for seed_node_addr in seed_node_addrs(network)
        .iter()
        .filter_map(SeedAddress::socket_addr)
    {
        if known_peers.try_get(rwtxn, &seed_node_addr)?.is_none() {
            known_peers.put(rwtxn, &seed_node_addr, &())?;
        }
    }
    Ok(())
}

pub async fn resolve_seed_address(
    dns_resolver: &TokioResolver,
    seed_addr: SeedAddress,
) -> Result<ResolvedSeedAddress, error::ResolveSeedAddress> {
    let domain = match seed_addr.host {
        url::Host::Ipv4(ipv4) => {
            return Ok(ResolvedSeedAddress::Static(SocketAddr::new(
                IpAddr::V4(ipv4),
                seed_addr.port,
            )));
        }
        url::Host::Ipv6(ipv6) => {
            return Ok(ResolvedSeedAddress::Static(SocketAddr::new(
                IpAddr::V6(ipv6),
                seed_addr.port,
            )));
        }
        url::Host::Domain(domain) => domain,
    };
    let mut addrs: Vec<_> = dns_resolver
        .lookup_ip(domain.as_str())
        .await
        .map_err(|err| error::ResolveSeedAddress::Net(Box::new(err)))?
        .into_iter()
        .filter(|addr| !addr.is_unspecified())
        .collect();
    let Some(last_addr) = addrs.pop() else {
        tracing::warn!(%domain, "the seed host name resolved to no address");
        return Err(error::ResolveSeedAddress::NoIpAddrs { domain });
    };
    addrs.reverse();
    Ok(ResolvedSeedAddress::Domain {
        domain,
        port: seed_addr.port,
        addrs: nonempty::NonEmpty {
            head: last_addr,
            tail: addrs,
        },
    })
}

/// Handle to the tasks that dial seed host names. Drop aborts the tasks.
#[repr(transparent)]
pub struct DialSeedsHandle(
    tokio_util::task::JoinMap<SeedAddress, Result<bool, error::DialSeed>>,
);

// Keep track of peer state
// Exchange metadata
// Bulk download
// Propagation
//
// Initial block download
//
// 1. Download headers
// 2. Download blocks
// 3. Update the state
#[derive(Clone)]
pub struct Net {
    pub server: Endpoint,
    archive: Archive,
    pub dns_resolver: Arc<TokioResolver>,
    magic_bytes: peer_message::MagicBytes,
    state: State,
    active_peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectionHandle>>>,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: DatabaseUnique<SerdeBincode<SocketAddr>, Unit>,
    /// Seed host names, which resolve at each dial
    seed_names: Arc<HashSet<SeedAddress>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Net {
    pub const NUM_DBS: u32 = 2;

    fn add_active_peer(
        &self,
        addr: SocketAddr,
        peer_connection_handle: PeerConnectionHandle,
        info_rx: mpsc::UnboundedReceiver<PeerConnectionInfo>,
    ) -> Result<(), error::AlreadyConnected> {
        tracing::trace!(%addr, "adding to active peers");
        let mut active_peers_write = self.active_peers.write();
        match active_peers_write.entry(addr) {
            hash_map::Entry::Occupied(_) => {
                tracing::error!(%addr, "already connected");
                return Err(error::AlreadyConnected(addr));
            }
            hash_map::Entry::Vacant(active_peer_entry) => {
                active_peer_entry.insert(peer_connection_handle);
            }
        }
        drop(active_peers_write);
        tokio::spawn({
            let info_rx = StreamNotifyClose::new(info_rx)
                .map(move |info| Ok((addr, info)));
            let peer_info_tx = self.peer_info_tx.clone();
            async move {
                if let Err(_send_err) = info_rx.forward(peer_info_tx).await {
                    tracing::error!(%addr, "Failed to send peer connection info");
                }
            }
        });
        Ok(())
    }

    pub fn remove_active_peer(&self, addr: SocketAddr) {
        tracing::trace!(%addr, "removing active peer");
        let mut active_peers_write = self.active_peers.write();
        if let Some(peer_connection) = active_peers_write.remove(&addr) {
            drop(peer_connection);
            tracing::info!(%addr, "disconnected");
        }
    }

    /// Apply the provided function to the peer connection handle,
    /// if it exists.
    pub fn try_with_active_peer_connection<F, T>(
        &self,
        addr: SocketAddr,
        f: F,
    ) -> Option<T>
    where
        F: FnMut(&PeerConnectionHandle) -> T,
    {
        let active_peers_read = self.active_peers.read();
        active_peers_read.get(&addr).map(f)
    }

    // TODO: This should have more context.
    // Last received message, connection state, etc.
    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.active_peers
            .read()
            .iter()
            .map(|(addr, conn_handle)| Peer {
                address: *addr,
                status: conn_handle.connection_status(),
            })
            .collect()
    }

    #[instrument(skip_all, fields(addr), err(Debug))]
    pub fn connect_peer(
        &self,
        env: sneed::Env<heed::WithoutTls>,
        mut resolved_addr: ResolvedSeedAddress,
    ) -> Result<(), Error> {
        {
            let active_peers = self.active_peers.read();
            for ip_addr in resolved_addr.ip_addrs() {
                let addr = SocketAddr::new(ip_addr, resolved_addr.port());
                if active_peers.contains_key(&addr) {
                    tracing::error!("already connected");
                    return Err(error::AlreadyConnected(addr).into());
                }
            }
        }
        let (addr, connecting) = loop {
            let addr = SocketAddr::new(
                resolved_addr.first_ip_addr(),
                resolved_addr.port(),
            );
            // This check happens within Quinn with a
            // generic "invalid remote address". We run the
            // same check, and provide a friendlier error
            // message.
            if addr.ip().is_unspecified() {
                return Err(Error::UnspecfiedPeerIP(addr.ip()));
            }
            let server_name = match resolved_addr.host() {
                url::Host::Domain(domain) => domain,
                url::Host::Ipv4(_) | url::Host::Ipv6(_) => "localhost",
            };
            match self.server.connect(addr, server_name) {
                Ok(connecting) => break (addr, connecting),
                Err(err @ quinn::ConnectError::InvalidRemoteAddress(_)) => {
                    let (_, Some(next_addr)) =
                        resolved_addr.pop_first_ip_addr()
                    else {
                        return Err(err.into());
                    };
                    resolved_addr = next_addr;
                }
                Err(err) => return Err(err.into()),
            }
        };
        // A host name resolves again at each start, so only an IP address
        // goes into the database.
        if let ResolvedSeedAddress::Static(static_addr) = resolved_addr {
            let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
            self.known_peers
                .put(&mut rwtxn, &static_addr, &())
                .map_err(DbError::from)?;
            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
            resolved_address: resolved_addr,
            state: self.state.clone(),
        };
        let (connection_handle, info_rx) =
            peer::connect(connecting, connection_ctxt);
        self.add_active_peer(addr, connection_handle, info_rx)?;
        Ok(())
    }

    /// Delete peer from known_peers DB.
    /// Connections to the peer are not terminated.
    pub fn forget_peer(
        &self,
        rwtxn: &mut RwTxn,
        addr: &SocketAddr,
    ) -> Result<bool, Error> {
        self.known_peers
            .delete(rwtxn, addr)
            .map_err(|err| DbError::from(err).into())
    }

    fn known_peer_addrs(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<SocketAddr>, DbError> {
        let peer_addrs = self.known_peers.iter_keys(rotxn)?.collect()?;
        Ok(peer_addrs)
    }

    fn is_active_peer(&self, resolved_addr: &ResolvedSeedAddress) -> bool {
        let active_peers = self.active_peers.read();
        resolved_addr.ip_addrs().any(|ip_addr| {
            active_peers
                .contains_key(&SocketAddr::new(ip_addr, resolved_addr.port()))
        })
    }

    /// Dial a peer that the node does not hold a connection to.
    /// Returns `true` if a connection started, and `false` if the peer is
    /// already connected.
    fn dial_peer(
        &self,
        env: sneed::Env<heed::WithoutTls>,
        resolved_addr: ResolvedSeedAddress,
    ) -> Result<bool, Error> {
        if self.is_active_peer(&resolved_addr) {
            return Ok(false);
        }
        let () = self.connect_peer(env, resolved_addr)?;
        Ok(true)
    }

    async fn dial_seed(
        &self,
        env: sneed::Env<heed::WithoutTls>,
        seed_addr: SeedAddress,
    ) -> Result<bool, error::DialSeed> {
        tracing::trace!(%seed_addr, "dial seed host name");
        let resolved_addr =
            resolve_seed_address(&self.dns_resolver, seed_addr).await?;
        self.dial_peer(env, resolved_addr)
            .map_err(|err| error::DialSeed::Connect(Box::new(err)))
    }

    /// Dial every address that the database holds, and every seed host name.
    /// A seed host name resolves again here, so a seed that moved to another
    /// address still gets a dial.
    /// Returns the number of connections that started.
    async fn dial_known_peers(
        &self,
        env: &sneed::Env<heed::WithoutTls>,
    ) -> Result<usize, Error> {
        let peer_addrs = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            self.known_peer_addrs(&rotxn)?
        };
        let mut dialed = 0;
        for peer_addr in peer_addrs {
            match self.dial_peer(env.clone(), peer_addr.into()) {
                Ok(true) => dialed += 1,
                Ok(false) => (),
                Err(err) => {
                    tracing::error!(%peer_addr, message = %ErrorChain::new(&err))
                }
            }
        }
        for seed_addr in self.seed_names.iter() {
            match self.dial_seed(env.clone(), seed_addr.clone()).await {
                Ok(true) => dialed += 1,
                Ok(false) => (),
                Err(err) => {
                    tracing::error!(%seed_addr, message = %ErrorChain::new(&err))
                }
            }
        }
        Ok(dialed)
    }

    /// Dial the known peers again while no peer connection exists.
    /// `min_delay` is the shortest wait between two checks for a connection.
    /// The wait doubles after each redial, up to `max_delay`.
    /// The future returns only on a database error.
    pub async fn redial_known_peers(
        &self,
        env: sneed::Env<heed::WithoutTls>,
        min_delay: Duration,
        max_delay: Duration,
    ) -> Result<(), Error> {
        let mut delay = min_delay;
        let mut no_peers_at_last_check = false;
        loop {
            tokio::time::sleep(delay).await;
            if !self.active_peers.read().is_empty() {
                delay = min_delay;
                no_peers_at_last_check = false;
                continue;
            }
            // The net task reconnects to a peer that errored. A redial waits
            // for a full delay with no connection, so it never dials first.
            if !no_peers_at_last_check {
                no_peers_at_last_check = true;
                continue;
            }
            let dialed = self.dial_known_peers(&env).await?;
            tracing::info!(dialed, "no peer connection: dialed known peers");
            delay = (2 * delay).min(max_delay);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &tokio::runtime::Handle,
        env: &sneed::Env<heed::WithoutTls>,
        archive: Archive,
        magic_bytes_override: Option<peer_message::MagicBytes>,
        network: Network,
        state: State,
        bind_addr: SocketAddr,
        add_peers: HashSet<SeedAddress>,
        server_names: HashSet<String>,
    ) -> Result<(Self, PeerInfoRx, DialSeedsHandle), Error> {
        let (server, _) = make_server_endpoint(bind_addr, server_names)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        let known_peers =
            match DatabaseUnique::open(env, &rwtxn, "known_peers")? {
                Some(known_peers) => known_peers,
                None => DatabaseUnique::create(env, &mut rwtxn, "known_peers")?,
            };
        let () = ensure_seed_peers(&known_peers, &mut rwtxn, network)?;
        for peer_addr in add_peers.iter().filter_map(SeedAddress::socket_addr) {
            known_peers.put(&mut rwtxn, &peer_addr, &())?;
        }
        let version = DatabaseUnique::create(env, &mut rwtxn, "net_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        let magic_bytes = magic_bytes_override
            .unwrap_or_else(|| peer_message::magic_bytes(network));
        let dns_resolver = {
            let builder = hickory_resolver::Resolver::builder_tokio()
                .map_err(Error::BuildDnsResolver)?;
            let resolver = builder.build().map_err(Error::BuildDnsResolver)?;
            Arc::new(resolver)
        };
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let seed_names: HashSet<SeedAddress> = seed_node_addrs(network)
            .iter()
            .map(|seed_addr| seed_addr.to_owned())
            .chain(add_peers)
            .filter(|seed_addr| seed_addr.socket_addr().is_none())
            .collect();
        let net = Net {
            server,
            archive,
            dns_resolver,
            magic_bytes,
            state,
            active_peers,
            peer_info_tx,
            known_peers,
            seed_names: Arc::new(seed_names),
            _version: version,
        };
        let known_peers = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            net.known_peer_addrs(&rotxn)?
        };
        let () = known_peers.into_iter().try_for_each(|peer_addr| {
            tracing::trace!(%peer_addr, "connecting to already known peer");
            match net.connect_peer(env.clone(), peer_addr.into()) {
                Err(Error::Connect(
                    quinn::ConnectError::InvalidRemoteAddress(addr),
                )) => {
                    tracing::warn!(
                        %addr, "new net: known peer with invalid remote address, removing"
                    );
                    let mut rwtxn = env.write_txn()?;
                    net.known_peers.delete(&mut rwtxn, &peer_addr).map_err(DbError::from)?;
                    rwtxn.commit()?;
                    tracing::info!(
                        %addr,
                        "new net: removed known peer with invalid remote address"
                    );
                    Ok(())
                }
                res => res,
            }
        })
        // TODO: would be better to indicate this in the return error?
        .inspect_err(|err| {
            tracing::error!("unable to connect to known peers during net construction: {err:#}");
        })?;
        let mut dial_seeds = tokio_util::task::JoinMap::new();
        for seed_addr in net.seed_names.iter().cloned() {
            let env = env.clone();
            let net = net.clone();
            dial_seeds.spawn_on(
                seed_addr.clone(),
                async move {
                    net.dial_seed(env, seed_addr).await.inspect_err(
                        |err| tracing::error!(message = %ErrorChain::new(err)),
                    )
                },
                runtime,
            );
        }
        Ok((net, peer_info_rx, DialSeedsHandle(dial_seeds)))
    }

    /// Accept the next incoming connection. Returns Some(addr) if a connection was accepted
    /// and a new peer was added.
    pub async fn accept_incoming(
        &self,
        env: sneed::Env<heed::WithoutTls>,
    ) -> Result<Option<SocketAddr>, error::AcceptConnection> {
        tracing::debug!(
            "listening for connections on `{}`",
            self.server
                .local_addr()
                .map(|socket| socket.to_string())
                .unwrap_or("unknown address".into())
        );
        let connection = match self.server.accept().await {
            Some(conn) => {
                let remote_address = conn.remote_address();
                tracing::trace!(%remote_address, "accepting connection");
                let raw_conn = conn.await.map_err(|error| {
                    error::AcceptConnection::Connection {
                        error,
                        remote_address,
                    }
                })?;
                Connection::new(raw_conn, self.magic_bytes)
            }
            None => {
                tracing::debug!("server endpoint closed");
                return Err(error::AcceptConnection::ServerEndpointClosed);
            }
        };
        let addr = connection.addr();
        tracing::trace!(%addr, "accepted incoming connection");
        if self.active_peers.read().contains_key(&addr) {
            tracing::info!(
                %addr, "already peered, refusing duplicate",
            );
            connection
                .inner
                .close(quinn::VarInt::from_u32(1), b"already connected");
        }
        if connection.inner.close_reason().is_some() {
            return Ok(None);
        }
        tracing::info!(%addr, "connected to new peer");
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        self.known_peers
            .put(&mut rwtxn, &addr, &())
            .map_err(DbError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        tracing::trace!(%addr, "wrote peer to database");
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
            resolved_address: addr.into(),
            state: self.state.clone(),
        };
        let (connection_handle, info_rx) =
            peer::handle(connection_ctxt, connection);
        self.add_active_peer(addr, connection_handle, info_rx)?;
        Ok(Some(addr))
    }

    /// Attempt to push an internal message to the specified peer
    /// Returns `true` if successful
    pub fn push_internal_message(
        &self,
        message: PeerConnectionMessage,
        addr: SocketAddr,
    ) -> bool {
        let active_peers_read = self.active_peers.read();
        let Some(peer_connection_handle) = active_peers_read.get(&addr) else {
            let err = Error::MissingPeerConnection(addr);
            tracing::warn!("{:#}", ErrorChain::new(&err));
            return false;
        };

        if let Err(send_err) = peer_connection_handle
            .internal_message_tx
            .unbounded_send(message)
        {
            let message = send_err.into_inner();
            tracing::warn!(
                "Failed to push internal message to peer connection {addr}: {message:?}"
            );
            return false;
        }
        true
    }

    /// Push a tx to all active peers, except those in the provided set
    pub fn push_tx(
        &self,
        exclude: HashSet<SocketAddr>,
        tx: &AuthorizedTransaction,
    ) {
        self.active_peers
            .read()
            .iter()
            .filter(|(addr, _)| !exclude.contains(addr))
            .for_each(|(addr, peer_connection_handle)| {
                match peer_connection_handle.connection_status() {
                    PeerConnectionStatus::Connecting => {
                        tracing::trace!(%addr, "skipping peer at {addr} because it is not fully connected");
                        return;
                    }
                    PeerConnectionStatus::Connected => {}
                }
                let request: PeerRequest = peer::message::PushTransactionRequest {
                    transaction: tx.clone(),
                }.into();
                if let Err(_send_err) = peer_connection_handle
                    .internal_message_tx
                    .unbounded_send(request.into())
                {
                    let txid = tx.transaction.txid();
                    tracing::warn!("Failed to push tx {txid} to peer at {addr}")
                }
            })
    }
}
