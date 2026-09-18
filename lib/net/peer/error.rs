#![allow(clippy::duplicated_attributes)]
use thiserror::Error;

use crate::net::peer::PeerStateId;

pub(in crate::net::peer) mod connection {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum Send {
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error("connection already closed")]
        ClosedStream(#[from] quinn::ClosedStream),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("write error ({stream_id})")]
        Write {
            stream_id: quinn::StreamId,
            source: quinn::WriteError,
        },
    }

    #[derive(Debug, Error)]
    #[error("Failed to send heartbeat")]
    #[repr(transparent)]
    pub struct SendHeartbeat(#[source] Send);

    impl<E> From<E> for SendHeartbeat
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to send request")]
    #[repr(transparent)]
    pub struct SendRequest(#[source] Send);

    impl<E> From<E> for SendRequest
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to send response")]
    pub struct SendResponse(#[source] Send);

    impl<E> From<E> for SendResponse
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    #[derive(Debug, Error)]
    pub enum SendMessage {
        #[error(transparent)]
        Heartbeat(#[from] SendHeartbeat),
        #[error(transparent)]
        Request(#[from] SendRequest),
    }

    #[derive(Debug, Error)]
    pub enum Receive {
        #[error("received incorrect magic: {}", const_hex::encode(.0))]
        BadMagic(crate::net::peer::message::MagicBytes),
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("failed to read magic bytes")]
        ReadMagic(#[source] quinn::ReadExactError),
        #[error("read to end error")]
        ReadToEnd(#[from] quinn::ReadToEndError),
        #[error("timed out waiting for response")]
        Timeout,
    }

    impl Receive {
        /// True when the peer sent another network's magic bytes.
        pub fn is_bad_magic(&self) -> bool {
            matches!(self, Self::BadMagic(_))
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to receive request from peer")]
    #[repr(transparent)]
    pub struct ReceiveRequest(#[source] Receive);

    impl ReceiveRequest {
        pub fn is_bad_magic(&self) -> bool {
            self.0.is_bad_magic()
        }
    }

    impl<E> From<E> for ReceiveRequest
    where
        Receive: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to receive response from peer")]
    #[repr(transparent)]
    pub struct ReceiveResponse(#[source] Receive);

    impl ReceiveResponse {
        pub fn is_bad_magic(&self) -> bool {
            self.0.is_bad_magic()
        }
    }

    impl<E> From<E> for ReceiveResponse
    where
        Receive: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }
}

pub(in crate::net::peer) mod channel_pool {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum Task {
        #[error("Send heartbeat task error")]
        Heartbeat(#[source] tokio::task::JoinError),
        #[error("Send request task error")]
        Request(#[source] tokio::task::JoinError),
    }

    #[allow(clippy::duplicated_attributes)]
    #[derive(transitive::Transitive, Debug, Error)]
    #[transitive(
        from(super::connection::SendHeartbeat, super::connection::SendMessage),
        from(super::connection::SendRequest, super::connection::SendMessage)
    )]
    pub enum SendMessage {
        #[error(transparent)]
        Connection(#[from] super::connection::SendMessage),
        #[error(transparent)]
        Task(#[from] Task),
    }

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send heartbeat message: receiver dropped")]
    pub struct SpawnHeartbeatTask;

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send request message: receiver dropped")]
    pub struct SpawnRequestTask;

    #[derive(Debug, Error)]
    pub enum SpawnTask {
        #[error(transparent)]
        Heartbeat(#[from] SpawnHeartbeatTask),
        #[error(transparent)]
        Request(#[from] SpawnRequestTask),
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error(transparent)]
        SendMessage(#[from] SendMessage),
        #[error(transparent)]
        SpawnTask(#[from] SpawnTask),
    }
}

pub(in crate::net::peer) mod request_queue {
    use thiserror::Error;

    #[derive(Debug, Error)]
    #[error("Failed to add heartbeat to send queue")]
    pub struct SendHeartbeat;

    #[derive(Debug, Error)]
    #[error("Failed to add request to send queue")]
    pub struct SendRequest;

    #[allow(clippy::duplicated_attributes)]
    #[derive(transitive::Transitive, Debug, Error)]
    #[transitive(
        from(super::channel_pool::SendMessage, super::channel_pool::Error),
        from(
            super::channel_pool::SpawnHeartbeatTask,
            super::channel_pool::SpawnTask
        ),
        from(
            super::channel_pool::SpawnRequestTask,
            super::channel_pool::SpawnTask
        ),
        from(super::channel_pool::SpawnTask, super::channel_pool::Error)
    )]
    pub enum Error {
        #[error(transparent)]
        ChannelPool(#[from] super::channel_pool::Error),
        #[error("Failed to push peer response")]
        PushPeerResponse,
    }
}

pub(in crate::net::peer) mod blocking_task {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum TaskError {
        #[error("archive error")]
        Archive(#[from] crate::archive::Error),
        #[error("peer should be banned; {0}")]
        PeerBan(#[from] crate::net::peer::BanReason),
        #[error(transparent)]
        ReadTxn(#[from] sneed::env::error::ReadTxn),
        #[error("Failed to push info message")]
        SendInfo,
        #[error(transparent)]
        SendRequest(#[from] super::request_queue::SendRequest),
        #[error("state error")]
        State(#[from] crate::state::Error),
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error("Failed to execute blocking task to completion")]
        Join(#[from] tokio::task::JoinError),
        #[error(transparent)]
        Task(Box<TaskError>),
    }

    impl From<TaskError> for Error {
        fn from(err: TaskError) -> Self {
            Self::Task(Box::new(err))
        }
    }
}

pub(in crate::net::peer) mod forward_response {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum TaskError {
        #[error("archive error")]
        Archive(#[source] Box<crate::archive::Error>),
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error(transparent)]
        ReadTxn(#[from] sneed::env::error::ReadTxn),
    }

    impl From<crate::archive::Error> for TaskError {
        fn from(err: crate::archive::Error) -> Self {
            Self::Archive(Box::new(err))
        }
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error("Failed to execute task to completion")]
        Join(#[from] tokio::task::JoinError),
        #[error(transparent)]
        Task(#[from] TaskError),
    }
}

pub mod mailbox {
    #[derive(thiserror::Error, Debug)]
    pub enum Error {
        #[error("Blocking task error")]
        BlockingTask(#[from] super::blocking_task::Error),
        #[error("Failed to generate response")]
        ForwardResponse(#[from] super::forward_response::Error),
        #[error("Heartbeat timeout")]
        HeartbeatTimeout,
        #[error("Failed to send response")]
        JoinSendResponse(#[source] tokio::task::JoinError),
        #[error(transparent)]
        ReceiveRequest(#[from] super::connection::ReceiveRequest),
        #[error(transparent)]
        RequestQueue(#[from] super::request_queue::Error),
        #[error(transparent)]
        SendResponse(#[from] super::connection::SendResponse),
    }

    impl Error {
        pub fn is_bad_magic(&self) -> bool {
            match self {
                Self::ReceiveRequest(err) => err.is_bad_magic(),
                _ => false,
            }
        }
    }
}

#[derive(Debug, Error)]
#[must_use]
pub enum Error {
    #[error("archive error")]
    Archive(#[from] crate::archive::Error),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("Database env error")]
    DbEnv(#[from] sneed::env::Error),
    #[error(transparent)]
    Mailbox(#[from] mailbox::Error),
    #[error("missing peer state for id {0}")]
    MissingPeerState(PeerStateId),
    #[error(transparent)]
    ReceiveResponse(#[from] connection::ReceiveResponse),
    #[error("Failed to push blocking task")]
    SendBlockingTask,
    #[error(transparent)]
    SendHeartbeat(#[from] request_queue::SendHeartbeat),
    #[error("Failed to push info message")]
    SendInfo,
    #[error(transparent)]
    SendRequest(#[from] request_queue::SendRequest),
    #[error(transparent)]
    SendResponse(#[from] connection::SendResponse),
    #[error("state error: {0}")]
    State(#[from] Box<crate::state::Error>),
}

impl From<crate::state::Error> for Error {
    fn from(err: crate::state::Error) -> Self {
        Self::State(Box::new(err))
    }
}

impl Error {
    pub fn is_connect_timeout(&self) -> bool {
        matches!(self, Self::Connection(quinn::ConnectionError::TimedOut))
    }

    pub fn is_duplicate_connection(&self) -> bool {
        let mut source = std::error::Error::source(self);
        while let Some(error) = source {
            if let Some(quinn::ConnectionError::ApplicationClosed(close)) =
                error.downcast_ref::<quinn::ConnectionError>()
            {
                return close.error_code == 1_u32.into()
                    && close.reason == b"already connected"[..];
            }
            source = error.source();
        }
        false
    }

    /// True when the peer answered with another network's magic bytes. Such a
    /// peer runs a different chain, so it never becomes useful.
    pub fn is_bad_magic(&self) -> bool {
        match self {
            Self::Mailbox(err) => err.is_bad_magic(),
            Self::ReceiveResponse(err) => err.is_bad_magic(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Error, connection, mailbox};
    use crate::net::peer::message;

    #[test]
    fn only_the_handshake_timeout_is_a_connect_timeout() {
        assert!(
            Error::Connection(quinn::ConnectionError::TimedOut)
                .is_connect_timeout()
        );
        for error in [
            Error::Mailbox(mailbox::Error::HeartbeatTimeout),
            Error::ReceiveResponse(connection::ReceiveResponse::from(
                connection::Receive::BadMagic([0; 4]),
            )),
            Error::ReceiveResponse(connection::ReceiveResponse::from(
                quinn::ConnectionError::TimedOut,
            )),
            Error::Connection(quinn::ConnectionError::ApplicationClosed(
                quinn::ApplicationClose {
                    error_code: 1_u32.into(),
                    reason: b"already connected"[..].into(),
                },
            )),
        ] {
            assert!(!error.is_connect_timeout());
        }
    }

    #[test]
    fn duplicate_close_survives_the_request_path() {
        let close = quinn::ConnectionError::ApplicationClosed(
            quinn::ApplicationClose {
                error_code: 1_u32.into(),
                reason: b"already connected"[..].into(),
            },
        );
        let err = Error::Mailbox(mailbox::Error::ReceiveRequest(
            connection::ReceiveRequest::from(close),
        ));
        assert!(err.is_duplicate_connection());
    }

    #[test]
    fn duplicate_close_survives_the_response_path() {
        let close = quinn::ConnectionError::ApplicationClosed(
            quinn::ApplicationClose {
                error_code: 1_u32.into(),
                reason: b"already connected"[..].into(),
            },
        );
        let err =
            Error::ReceiveResponse(connection::ReceiveResponse::from(close));
        assert!(err.is_duplicate_connection());
    }

    #[test]
    fn other_connection_errors_are_not_duplicate_closes() {
        for error in [
            quinn::ConnectionError::TimedOut,
            quinn::ConnectionError::ApplicationClosed(
                quinn::ApplicationClose {
                    error_code: 0_u32.into(),
                    reason: b"already connected"[..].into(),
                },
            ),
            quinn::ConnectionError::ApplicationClosed(
                quinn::ApplicationClose {
                    error_code: 1_u32.into(),
                    reason: b"test complete"[..].into(),
                },
            ),
        ] {
            assert!(!Error::Connection(error).is_duplicate_connection());
        }
    }

    const FOREIGN_MAGIC: message::MagicBytes = [0x85, 0x18, 0x95, 0x01];

    // A peer that answers a request with another network's magic must read as
    // bad magic through every wrapper the error passes.
    #[test]
    fn bad_magic_survives_the_request_path() {
        let inner = connection::Receive::BadMagic(FOREIGN_MAGIC);
        let err = Error::Mailbox(mailbox::Error::ReceiveRequest(
            connection::ReceiveRequest::from(inner),
        ));
        assert!(err.is_bad_magic());
    }

    #[test]
    fn bad_magic_survives_the_response_path() {
        let inner = connection::Receive::BadMagic(FOREIGN_MAGIC);
        let err =
            Error::ReceiveResponse(connection::ReceiveResponse::from(inner));
        assert!(err.is_bad_magic());
    }

    // A timeout says nothing about the peer's network, so the node keeps it.
    #[test]
    fn a_timeout_is_not_bad_magic() {
        let err = Error::ReceiveResponse(connection::ReceiveResponse::from(
            connection::Receive::Timeout,
        ));
        assert!(!err.is_bad_magic());
    }

    #[test]
    fn a_heartbeat_timeout_is_not_bad_magic() {
        let err = Error::Mailbox(mailbox::Error::HeartbeatTimeout);
        assert!(!err.is_bad_magic());
    }
}
