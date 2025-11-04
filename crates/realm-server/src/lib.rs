//! Realm WebSocket Server
//!
//! WebSocket-based inference server inspired by Polkadot's parachain runtime model.
//! Uses function dispatch instead of traditional REST endpoints.

pub mod auth;
pub mod dispatcher;
pub mod metrics_server;
pub mod orchestrator;
pub mod pipeline_dsl;
pub mod protocol;
pub mod rate_limiter;
pub mod runtime_manager;
pub mod websocket;

use crate::auth::ApiKeyStore;
use crate::dispatcher::FunctionDispatcher;
use crate::metrics_server::{MetricsServer, MetricsServerConfig};
use crate::rate_limiter::{RateLimiter, RateLimiterConfig};
use crate::websocket::WebSocketHandler;
use anyhow::{Context, Result};
use realm_metrics::collector::MetricsCollector;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tracing::{error, info};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,

    /// Port to listen on
    pub port: u16,

    /// Enable authentication (requires API keys)
    pub enable_auth: bool,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Metrics server configuration (optional)
    pub metrics_config: Option<MetricsServerConfig>,

    /// API key store configuration
    pub api_key_store_config: Option<auth::ApiKeyStoreConfig>,

    /// Rate limiter configuration (optional)
    pub rate_limiter_config: Option<RateLimiterConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            enable_auth: false,
            max_connections: 1000,
            metrics_config: Some(MetricsServerConfig::default()),
            api_key_store_config: None,
            rate_limiter_config: Some(RateLimiterConfig::default()),
        }
    }
}

/// Realm WebSocket Server
pub struct RealmServer {
    /// Server configuration
    config: ServerConfig,

    /// Function dispatcher
    dispatcher: Arc<FunctionDispatcher>,

    /// Metrics collector
    metrics: Arc<MetricsCollector>,

    /// API key store (optional, when auth is enabled)
    api_key_store: Option<Arc<ApiKeyStore>>,

    /// Rate limiter (optional)
    rate_limiter: Option<Arc<RateLimiter>>,
}

impl RealmServer {
    /// Create a new Realm server with default dispatcher
    pub fn new(config: ServerConfig) -> Result<Self> {
        let api_key_store = if config.enable_auth {
            let store_config = config.api_key_store_config.clone().unwrap_or_default();
            Some(Arc::new(ApiKeyStore::new(store_config)?))
        } else {
            None
        };

        let rate_limiter = config
            .rate_limiter_config
            .clone()
            .map(|cfg| Arc::new(RateLimiter::new(cfg)));

        Ok(Self {
            config,
            dispatcher: Arc::new(FunctionDispatcher::new()),
            metrics: Arc::new(MetricsCollector::new()),
            api_key_store,
            rate_limiter,
        })
    }

    /// Create a server with a custom dispatcher (e.g., with runtime manager)
    pub fn with_dispatcher(config: ServerConfig, dispatcher: FunctionDispatcher) -> Result<Self> {
        let api_key_store = if config.enable_auth {
            let store_config = config.api_key_store_config.clone().unwrap_or_default();
            Some(Arc::new(ApiKeyStore::new(store_config)?))
        } else {
            None
        };

        let rate_limiter = config
            .rate_limiter_config
            .clone()
            .map(|cfg| Arc::new(RateLimiter::new(cfg)));

        Ok(Self {
            config,
            dispatcher: Arc::new(dispatcher),
            metrics: Arc::new(MetricsCollector::new()),
            api_key_store,
            rate_limiter,
        })
    }

    /// Get metrics collector
    pub fn metrics(&self) -> Arc<MetricsCollector> {
        self.metrics.clone()
    }

    /// Get the server address
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }

    /// Run the server
    pub async fn run(&self) -> Result<()> {
        let addr: SocketAddr = self.address().parse().context("Invalid server address")?;

        let listener = TcpListener::bind(&addr)
            .await
            .context("Failed to bind to address")?;

        info!("🚀 Realm WebSocket server listening on ws://{}", addr);
        info!("Runtime version: {}", env!("CARGO_PKG_VERSION"));

        // Print available functions
        let metadata = self.dispatcher.metadata();
        info!("Available functions:");
        for func in metadata.functions {
            info!("  - {} ({})", func.name, func.description);
        }

        // Start metrics server if configured
        if let Some(ref metrics_config) = self.config.metrics_config {
            let metrics_server = MetricsServer::new(metrics_config.clone(), self.metrics.clone());
            let metrics_addr = metrics_server.address();

            tokio::spawn(async move {
                if let Err(e) = metrics_server.run().await {
                    error!("Metrics server error: {}", e);
                }
            });

            info!("📊 Metrics available at http://{}/metrics", metrics_addr);
        }

        // Accept connections
        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    info!("New connection from {}", peer_addr);

                    let dispatcher = self.dispatcher.clone();
                    let config = self.config.clone();
                    let api_key_store = self.api_key_store.clone();
                    let rate_limiter = self.rate_limiter.clone();

                    // Spawn handler for this connection
                    tokio::spawn(async move {
                        match accept_async(stream).await {
                            Ok(ws) => {
                                let handler = if config.enable_auth {
                                    if let Some(store) = api_key_store {
                                        // Use authenticated handler with API key store
                                        match WebSocketHandler::with_auth(
                                            dispatcher,
                                            store,
                                            rate_limiter,
                                            ws,
                                        )
                                        .await
                                        {
                                            Ok(()) => {}
                                            Err(e) => {
                                                error!(
                                                    "Authenticated connection handler error: {}",
                                                    e
                                                );
                                            }
                                        }
                                        return;
                                    } else {
                                        // Fallback to default tenant if no store configured
                                        WebSocketHandler::with_tenant(
                                            dispatcher,
                                            "default",
                                            rate_limiter,
                                        )
                                    }
                                } else {
                                    WebSocketHandler::with_rate_limiter(dispatcher, rate_limiter)
                                };

                                if let Err(e) = handler.handle_connection(ws).await {
                                    error!("Connection handler error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("WebSocket handshake failed: {}", e);
                            }
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert!(!config.enable_auth);
        assert_eq!(config.max_connections, 1000);
    }

    #[test]
    fn test_server_address() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 9090,
            ..Default::default()
        };

        let server = RealmServer::new(config).unwrap();
        assert_eq!(server.address(), "0.0.0.0:9090");
    }

    #[test]
    fn test_server_creation() {
        let server = RealmServer::new(ServerConfig::default()).unwrap();
        assert_eq!(server.address(), "127.0.0.1:8080");
    }
}
