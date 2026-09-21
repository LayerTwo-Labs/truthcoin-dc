use std::{
    fmt::Display,
    net::{IpAddr, SocketAddr},
    str::FromStr,
};

use thiserror::Error;

use crate::types::THIS_SIDECHAIN;

pub const DEFAULT_PORT: u16 = 4000 + THIS_SIDECHAIN as u16;

#[derive(Debug, Error)]
#[error("cannot parse the seed address")]
pub struct ParseSeedAddressError(#[from] url::ParseError);

/// Address of a seed peer: a host name or an IP address, and a port
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SeedAddress<S = String> {
    pub host: url::Host<S>,
    pub port: u16,
}

impl<S> SeedAddress<S> {
    /// The socket address, if the host is an IP address
    pub fn socket_addr(&self) -> Option<SocketAddr> {
        match self.host {
            url::Host::Domain(_) => None,
            url::Host::Ipv4(ipv4) => {
                Some(SocketAddr::new(IpAddr::V4(ipv4), self.port))
            }
            url::Host::Ipv6(ipv6) => {
                Some(SocketAddr::new(IpAddr::V6(ipv6), self.port))
            }
        }
    }
}

impl SeedAddress<&str> {
    pub fn to_owned(&self) -> SeedAddress {
        SeedAddress {
            host: self.host.to_owned(),
            port: self.port,
        }
    }
}

impl<S> Display for SeedAddress<S>
where
    url::Host<S>: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { host, port } = self;
        write!(f, "{host}:{port}")
    }
}

/// Parses `host:port`, or `host` with the default port. Put an IPv6 address
/// in brackets, as in `[::1]:4004`.
impl FromStr for SeedAddress {
    type Err = ParseSeedAddressError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (host_str, port) = match s.rsplit_once(':') {
            Some((host_str, port)) => {
                let port: u16 =
                    port.parse().map_err(|_| url::ParseError::InvalidPort)?;
                (host_str, port)
            }
            None => (s, DEFAULT_PORT),
        };
        let host = url::Host::parse(host_str)?;
        Ok(Self { host, port })
    }
}

/// A seed address, resolved to one or more IP addresses
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedSeedAddress {
    Domain {
        domain: String,
        port: u16,
        /// Addresses in reverse order, so the last element is the first
        /// resolved address
        addrs: nonempty::NonEmpty<IpAddr>,
    },
    /// An IP address needs no resolution
    Static(SocketAddr),
}

impl ResolvedSeedAddress {
    pub fn host(&self) -> url::Host<&str> {
        match self {
            Self::Domain { domain, .. } => url::Host::Domain(domain),
            Self::Static(SocketAddr::V4(v4)) => url::Host::Ipv4(*v4.ip()),
            Self::Static(SocketAddr::V6(v6)) => url::Host::Ipv6(*v6.ip()),
        }
    }

    pub fn port(&self) -> u16 {
        match self {
            Self::Domain { port, .. } => *port,
            Self::Static(addr) => addr.port(),
        }
    }

    pub fn as_seed_address(&self) -> SeedAddress<&str> {
        SeedAddress {
            host: self.host(),
            port: self.port(),
        }
    }

    /// The first resolved IP address
    pub fn first_ip_addr(&self) -> IpAddr {
        match self {
            Self::Domain { addrs, .. } => *addrs.last(),
            Self::Static(addr) => addr.ip(),
        }
    }

    /// Removes the first resolved IP address. The second value holds the
    /// other addresses, if there are any.
    pub fn pop_first_ip_addr(self) -> (IpAddr, Option<Self>) {
        match self {
            Self::Domain {
                domain,
                port,
                mut addrs,
            } => match addrs.pop() {
                Some(addr) => (
                    addr,
                    Some(Self::Domain {
                        domain,
                        port,
                        addrs,
                    }),
                ),
                None => (addrs.head, None),
            },
            Self::Static(addr) => (addr.ip(), None),
        }
    }

    pub fn ip_addrs(&self) -> impl Iterator<Item = IpAddr> {
        match self {
            Self::Domain { addrs, .. } => Box::new(addrs.iter().rev().cloned())
                as Box<dyn Iterator<Item = IpAddr>>,
            Self::Static(addr) => Box::new(std::iter::once(addr.ip())),
        }
    }
}

impl From<SocketAddr> for ResolvedSeedAddress {
    fn from(addr: SocketAddr) -> Self {
        Self::Static(addr)
    }
}
