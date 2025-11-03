//! WebSocket Server Example
//!
//! Demonstrates running the Realm WebSocket server with metrics endpoint.

use anyhow::Result;
use realm_server::{RealmServer, ServerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .init();

    // Create server configuration
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        enable_auth: false,
        max_connections: 1000,
        metrics_config: Some(realm_server::metrics_server::MetricsServerConfig {
            host: "127.0.0.1".to_string(),
            port: 9090,
        }),
        api_key_store_config: None,
        rate_limiter_config: None,
    };

    // Create and run server
    let server = RealmServer::new(config)?;

    println!("🚀 Starting Realm WebSocket Server");
    println!("WebSocket: ws://127.0.0.1:8080");
    println!("Metrics: http://127.0.0.1:9090/metrics");
    println!();
    println!("Try connecting with:");
    println!("  wscat -c ws://127.0.0.1:8080");
    println!();
    println!("Or fetch metrics:");
    println!("  curl http://127.0.0.1:9090/metrics");
    println!();

    server.run().await
}
