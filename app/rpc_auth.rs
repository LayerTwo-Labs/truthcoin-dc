use std::{
    collections::HashSet,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::Arc,
    task::{Context, Poll},
};

use jsonrpsee::{
    server::middleware::rpc::{
        Batch, BatchEntry, BatchEntryErr, MethodResponse, Notification,
        ResponseFuture, RpcServiceT,
    },
    types::{ErrorObject, ErrorObjectOwned, Request},
};
use subtle::ConstantTimeEq as _;

pub const COOKIE_FILE_NAME: &str = ".cookie";

const UNAUTHORIZED_CODE: i32 = -32001;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RPC cookie file `{}` holds no valid token", .path.display())]
    Malformed { path: PathBuf },
    #[error("failed to read RPC cookie file `{}`", .path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write RPC cookie file `{}`", .path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl Error {
    fn write(path: &Path) -> impl FnOnce(io::Error) -> Self {
        let path = path.to_owned();
        move |source| Self::Write { path, source }
    }
}

fn write_cookie(path: &Path, token: &str) -> Result<(), Error> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(Error::write(dir))?;
    }
    let tmp_path = path.with_extension("tmp");
    {
        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file =
            options.open(&tmp_path).map_err(Error::write(&tmp_path))?;
        file.write_all(token.as_bytes())
            .map_err(Error::write(&tmp_path))?;
        file.sync_all().map_err(Error::write(&tmp_path))?;
    }
    fs::rename(&tmp_path, path).map_err(Error::write(path))
}

/// Returns the token in `path`, and writes a new random token if the file
/// does not exist.
pub fn load_or_create_cookie(path: &Path) -> Result<String, Error> {
    match fs::read_to_string(path) {
        Ok(content) => {
            let token = content.trim();
            if token.is_empty() || token.contains(char::is_whitespace) {
                return Err(Error::Malformed {
                    path: path.to_owned(),
                });
            }
            Ok(token.to_owned())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            let mut bytes = [0u8; 32];
            rand::fill(&mut bytes[..]);
            let token = const_hex::encode(bytes);
            write_cookie(path, &token)?;
            tracing::info!(path = %path.display(), "Created RPC cookie file");
            Ok(token)
        }
        Err(source) => Err(Error::Read {
            path: path.to_owned(),
            source,
        }),
    }
}

/// Request extension set when the bearer token matches the cookie.
#[derive(Clone, Copy, Debug)]
struct RpcAuthenticated;

#[derive(Clone, Debug)]
pub struct BearerAuth<S> {
    inner: S,
    token: Arc<str>,
}

/// Marks a request that carries the cookie token. It never refuses a
/// request, so public methods stay open; [`GuardLayer`] does the refusal.
#[derive(Clone, Debug)]
pub struct BearerAuthLayer {
    token: Arc<str>,
}

impl BearerAuthLayer {
    pub fn new(token: Arc<str>) -> Self {
        Self { token }
    }
}

impl<S> tower::Layer<S> for BearerAuthLayer {
    type Service = BearerAuth<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BearerAuth {
            inner,
            token: self.token.clone(),
        }
    }
}

impl<S, B> tower::Service<http::Request<B>> for BearerAuth<S>
where
    S: tower::Service<http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: http::Request<B>) -> Self::Future {
        let authenticated = request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.as_bytes().strip_prefix(b"Bearer "))
            .is_some_and(|token| {
                bool::from(token.ct_eq(self.token.as_bytes()))
            });
        if authenticated {
            request.extensions_mut().insert(RpcAuthenticated);
        }
        self.inner.call(request)
    }
}

#[derive(Clone, Debug)]
pub struct Guard<S> {
    inner: S,
    guarded_methods: Arc<HashSet<&'static str>>,
    unauthorized: ErrorObjectOwned,
}

/// Refuses a call to a guarded method unless [`BearerAuthLayer`] marked
/// the request.
#[derive(Clone, Debug)]
pub struct GuardLayer {
    guarded_methods: Arc<HashSet<&'static str>>,
    unauthorized: ErrorObjectOwned,
}

impl GuardLayer {
    pub fn new(
        guarded_methods: HashSet<&'static str>,
        cookie_path: &Path,
    ) -> Self {
        let unauthorized = ErrorObject::owned(
            UNAUTHORIZED_CODE,
            format!(
                "unauthorized: send Authorization: Bearer <token from {}>",
                cookie_path.display()
            ),
            None::<()>,
        );
        Self {
            guarded_methods: Arc::new(guarded_methods),
            unauthorized,
        }
    }
}

impl<S> tower::Layer<S> for GuardLayer {
    type Service = Guard<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Guard {
            inner,
            guarded_methods: self.guarded_methods.clone(),
            unauthorized: self.unauthorized.clone(),
        }
    }
}

impl<S> Guard<S> {
    fn refuses(&self, request: &Request<'_>) -> bool {
        self.guarded_methods.contains(request.method_name())
            && request.extensions().get::<RpcAuthenticated>().is_none()
    }
}

impl<S> RpcServiceT for Guard<S>
where
    S: RpcServiceT<
            MethodResponse = MethodResponse,
            BatchResponse = MethodResponse,
            NotificationResponse = MethodResponse,
        >,
{
    type MethodResponse = MethodResponse;
    type BatchResponse = MethodResponse;
    type NotificationResponse = MethodResponse;

    fn call<'a>(
        &self,
        request: Request<'a>,
    ) -> impl Future<Output = Self::MethodResponse> + Send + 'a {
        if self.refuses(&request) {
            ResponseFuture::ready(MethodResponse::error(
                request.id,
                self.unauthorized.clone(),
            ))
        } else {
            ResponseFuture::future(self.inner.call(request))
        }
    }

    fn batch<'a>(
        &self,
        mut batch: Batch<'a>,
    ) -> impl Future<Output = Self::BatchResponse> + Send + 'a {
        // The inner service runs batch calls itself, not through `call`.
        for entry in batch.iter_mut() {
            if let Ok(BatchEntry::Call(request)) = entry
                && self.refuses(request)
            {
                *entry = Err(BatchEntryErr::new(
                    request.id.clone(),
                    self.unauthorized.clone(),
                ));
            }
        }
        self.inner.batch(batch)
    }

    fn notification<'a>(
        &self,
        notification: Notification<'a>,
    ) -> impl Future<Output = Self::NotificationResponse> + Send + 'a {
        self.inner.notification(notification)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs, net::SocketAddr, path::Path};

    use jsonrpsee::{
        RpcModule,
        core::middleware::RpcServiceBuilder,
        server::{Server, ServerHandle},
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    use super::{
        BearerAuthLayer, COOKIE_FILE_NAME, Error, GuardLayer,
        load_or_create_cookie,
    };

    const TOKEN: &str = "0123456789abcdef";

    #[test]
    fn creates_a_private_cookie_and_reuses_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(COOKIE_FILE_NAME);
        let token = load_or_create_cookie(&path)?;
        assert_eq!(token.len(), 64);
        assert!(
            token
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(&path)?.permissions().mode() & 0o777,
                0o600
            );
        }
        assert_eq!(load_or_create_cookie(&path)?, token);
        Ok(())
    }

    #[test]
    fn refuses_an_empty_cookie() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(COOKIE_FILE_NAME);
        fs::write(&path, " \n")?;
        assert!(matches!(
            load_or_create_cookie(&path),
            Err(Error::Malformed { .. })
        ));
        Ok(())
    }

    async fn start_server() -> anyhow::Result<(SocketAddr, ServerHandle)> {
        let server = Server::builder()
            .set_http_middleware(
                tower::ServiceBuilder::new()
                    .layer(BearerAuthLayer::new(TOKEN.into())),
            )
            .set_rpc_middleware(RpcServiceBuilder::new().layer(
                GuardLayer::new(
                    HashSet::from(["secret"]),
                    Path::new("/datadir/.cookie"),
                ),
            ))
            .build(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await?;
        let addr = server.local_addr()?;
        let mut module = RpcModule::new(());
        module.register_method("ping", |_, _, _| "pong")?;
        module.register_method("secret", |_, _, _| "seed")?;
        Ok((addr, server.start(module)))
    }

    async fn post(
        addr: SocketAddr,
        authorization: Option<&str>,
        body: &str,
    ) -> anyhow::Result<String> {
        let authorization = authorization
            .map(|value| format!("Authorization: {value}\r\n"))
            .unwrap_or_default();
        let request = format!(
            "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\n{authorization}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = tokio::net::TcpStream::connect(addr).await?;
        stream.write_all(request.as_bytes()).await?;
        let mut response = String::new();
        stream.read_to_string(&mut response).await?;
        Ok(response)
    }

    fn call(method: &str) -> String {
        format!(r#"{{"jsonrpc":"2.0","id":1,"method":"{method}"}}"#)
    }

    const UNAUTHORIZED: &str = r#""code":-32001"#;

    #[tokio::test]
    async fn guarded_method_needs_the_token() -> anyhow::Result<()> {
        let (addr, handle) = start_server().await?;
        let bearer = format!("Bearer {TOKEN}");

        let response = post(addr, None, &call("secret")).await?;
        assert!(response.contains(UNAUTHORIZED), "{response}");

        let response = post(addr, Some(&bearer), &call("secret")).await?;
        assert!(response.contains(r#""result":"seed""#), "{response}");

        let response =
            post(addr, Some("Bearer 0000000000000000"), &call("secret"))
                .await?;
        assert!(response.contains(UNAUTHORIZED), "{response}");

        let batch = format!("[{},{}]", call("ping"), call("secret"));
        let response = post(addr, None, &batch).await?;
        assert!(response.contains(r#""result":"pong""#), "{response}");
        assert!(response.contains(UNAUTHORIZED), "{response}");

        handle.stop()?;
        Ok(())
    }

    #[tokio::test]
    async fn public_method_needs_no_token() -> anyhow::Result<()> {
        let (addr, handle) = start_server().await?;
        let response = post(addr, None, &call("ping")).await?;
        assert!(response.contains(r#""result":"pong""#), "{response}");
        handle.stop()?;
        Ok(())
    }
}
