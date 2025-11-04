//! HTTP Metrics Server
//!
//! Provides HTTP endpoint for Prometheus to scrape metrics.

use anyhow::{Context, Result};
use realm_metrics::collector::MetricsCollector;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};

/// HTTP metrics server configuration
#[derive(Debug, Clone)]
pub struct MetricsServerConfig {
    /// Host address to bind to
    pub host: String,

    /// Port to listen on
    pub port: u16,
}

impl Default for MetricsServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9090,
        }
    }
}

/// HTTP metrics server for Prometheus
pub struct MetricsServer {
    /// Configuration
    config: MetricsServerConfig,

    /// Metrics collector
    collector: Arc<MetricsCollector>,
}

impl MetricsServer {
    /// Create a new metrics server
    pub fn new(config: MetricsServerConfig, collector: Arc<MetricsCollector>) -> Self {
        Self { config, collector }
    }

    /// Get the server address
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }

    /// Run the metrics server
    pub async fn run(&self) -> Result<()> {
        let addr: SocketAddr = self
            .address()
            .parse()
            .context("Invalid metrics server address")?;

        let listener = TcpListener::bind(&addr)
            .await
            .context("Failed to bind metrics server")?;

        info!("📊 Metrics server listening on http://{}/metrics", addr);

        loop {
            match listener.accept().await {
                Ok((mut stream, peer_addr)) => {
                    debug!("Metrics request from {}", peer_addr);

                    // Read request
                    let mut buffer = [0u8; 1024];
                    match stream.read(&mut buffer).await {
                        Ok(n) if n > 0 => {
                            let request = String::from_utf8_lossy(&buffer[..n]);
                            debug!("Request: {}", request.lines().next().unwrap_or(""));

                            // Check if it's a GET /metrics request
                            if request.starts_with("GET /metrics") {
                                // Export all metrics in Prometheus format
                                let metrics_text = self.collector.export_prometheus();

                                if !metrics_text.is_empty() {
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\n\
                                         Content-Type: text/plain; version=0.0.4\r\n\
                                         Content-Length: {}\r\n\
                                         \r\n\
                                         {}",
                                        metrics_text.len(),
                                        metrics_text
                                    );

                                    if let Err(e) = stream.write_all(response.as_bytes()).await {
                                        error!("Failed to write response: {}", e);
                                    }
                                } else {
                                    // No metrics yet
                                    let empty_response = "HTTP/1.1 200 OK\r\n\
                                                        Content-Type: text/plain; version=0.0.4\r\n\
                                                        Content-Length: 0\r\n\
                                                        \r\n";

                                    let _ = stream.write_all(empty_response.as_bytes()).await;
                                }
                            } else {
                                // Return 404 for other paths
                                let response = "HTTP/1.1 404 Not Found\r\n\
                                              Content-Type: text/plain\r\n\
                                              Content-Length: 23\r\n\
                                              \r\n\
                                              Only /metrics is supported";

                                if let Err(e) = stream.write_all(response.as_bytes()).await {
                                    error!("Failed to write 404 response: {}", e);
                                }
                            }
                        }
                        Ok(_) => {
                            warn!("Empty request from {}", peer_addr);
                        }
                        Err(e) => {
                            error!("Failed to read request: {}", e);
                        }
                    }

                    // Close connection
                    let _ = stream.shutdown().await;
                }
                Err(e) => {
                    error!("Failed to accept metrics connection: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_server_config_default() {
        let config = MetricsServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9090);
    }

    #[test]
    fn test_metrics_server_address() {
        let config = MetricsServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
        };

        let collector = Arc::new(MetricsCollector::new());
        let server = MetricsServer::new(config, collector);
        assert_eq!(server.address(), "0.0.0.0:8080");
    }
}
