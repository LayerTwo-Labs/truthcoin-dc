//! Convenience and utility types and functions

use std::{marker::Unpin, task::Poll};

use futures::{Stream, StreamExt};
use poll_promise::Promise;
use tokio::runtime::Handle;

type PromiseStreamInner<S> = Promise<Option<(S, <S as Stream>::Item)>>;

pub struct PromiseStream<S>
where
    S: Send + Stream + 'static,
    S::Item: Send,
{
    inner: Option<PromiseStreamInner<S>>,
    handle: Handle,
}

impl<S> PromiseStream<S>
where
    S: Send + Stream + Unpin + 'static,
    S::Item: Send,
{
    pub fn new(stream: S, handle: Handle) -> Self {
        let inner = Some(Self::spawn(stream, &handle));
        Self { inner, handle }
    }

    fn spawn(mut stream: S, handle: &Handle) -> PromiseStreamInner<S> {
        let (sender, promise) = Promise::new();
        handle.spawn(async move {
            sender.send(stream.next().await.map(|item| (stream, item)));
        });
        promise
    }

    /// Get the next item in the stream if available,
    /// or return None if the stream is complete
    pub fn poll_next(&mut self) -> Option<Poll<S::Item>> {
        let res;
        let inner = std::mem::take(&mut self.inner);
        self.inner = match inner {
            Some(promise) => match promise.try_take() {
                Ok(Some((stream, item))) => {
                    res = Some(Poll::Ready(item));
                    Some(Self::spawn(stream, &self.handle))
                }
                Ok(None) => {
                    res = None;
                    None
                }
                Err(promise) => {
                    res = Some(Poll::Pending);
                    Some(promise)
                }
            },
            None => {
                res = None;
                None
            }
        };
        res
    }
}

/// Saturating predecessor of a log level
pub fn saturating_pred_level(log_level: tracing::Level) -> tracing::Level {
    match log_level {
        tracing::Level::TRACE => tracing::Level::DEBUG,
        tracing::Level::DEBUG => tracing::Level::INFO,
        tracing::Level::INFO => tracing::Level::WARN,
        tracing::Level::WARN => tracing::Level::ERROR,
        tracing::Level::ERROR => tracing::Level::ERROR,
    }
}
