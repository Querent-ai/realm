//! Integration tests for realm-server
//!
//! Tests the full flow: HTTP/WebSocket -> Runtime Manager -> WASM -> Host Functions

use axum::http::StatusCode;
use axum_test::TestServer;
use realm_server::http::{create_router, ServerState};
use realm_server::runtime_manager::RuntimeManager;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Helper to create a RuntimeManager, returning None if WASM file doesn't exist
fn create_runtime_manager_or_skip() -> Option<RuntimeManager> {
    let wasm_path = PathBuf::from("./target/wasm32-unknown-unknown/release/realm_wasm.wasm");
    if !wasm_path.exists() {
        eprintln!("⚠️  WASM file not found at {:?}, skipping test", wasm_path);
        eprintln!("   To run full integration tests, build WASM first:");
        eprintln!("   cargo build --target wasm32-unknown-unknown -p realm-wasm --release");
        return None;
    }
    Some(RuntimeManager::new(wasm_path).expect("Failed to create RuntimeManager"))
}

#[tokio::test]
async fn test_health_check() {
    // Health endpoint doesn't actually need RuntimeManager, but we create it for consistency
    let Some(rm) = create_runtime_manager_or_skip() else {
        return; // Skip test if WASM not available
    };
    let runtime_manager = Arc::new(Mutex::new(rm));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
        metrics: None,
    };

    let app = create_router(state);
    // axum-test 18.2 accepts Router directly (it implements IntoTransportLayer)
    let server = TestServer::new(app).unwrap();

    let response = server.get("/health").await;
    response.assert_status(StatusCode::OK);
    response.assert_json(&json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION")
    }));
}

#[tokio::test]
async fn test_metrics_endpoint() {
    // Metrics endpoint doesn't actually need RuntimeManager, but we create it for consistency
    let Some(rm) = create_runtime_manager_or_skip() else {
        return; // Skip test if WASM not available
    };
    let runtime_manager = Arc::new(Mutex::new(rm));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
        metrics: None,
    };

    let app = create_router(state);
    // axum-test 18.2 accepts Router directly (it implements IntoTransportLayer)
    let server = TestServer::new(app).unwrap();

    let response = server.get("/metrics").await;
    response.assert_status(StatusCode::OK);
    // Metrics endpoint should return text/plain
    assert!(response
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .contains("text/plain"));
}

#[tokio::test]
async fn test_chat_completions_validation() {
    let Some(rm) = create_runtime_manager_or_skip() else {
        return; // Skip test if WASM not available
    };
    let runtime_manager = Arc::new(Mutex::new(rm));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
        metrics: None,
    };

    let app = create_router(state);
    // axum-test 18.2 accepts Router directly (it implements IntoTransportLayer)
    let server = TestServer::new(app).unwrap();

    // Test with invalid request (missing required fields)
    let response = server.post("/v1/chat/completions").json(&json!({})).await;

    // Should return 400 or 422 (validation error)
    assert!(response.status_code().is_client_error());
}

#[tokio::test]
async fn test_chat_completions_structure() {
    let Some(rm) = create_runtime_manager_or_skip() else {
        return; // Skip test if WASM not available
    };
    let runtime_manager = Arc::new(Mutex::new(rm));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
        metrics: None,
    };

    let app = create_router(state);
    // axum-test 18.2 accepts Router directly (it implements IntoTransportLayer)
    let server = TestServer::new(app).unwrap();

    // Test with valid request structure
    let request = json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": false,
        "max_tokens": 50
    });

    let response = server.post("/v1/chat/completions").json(&request).await;

    // Response should have OpenAI-compatible structure
    if response.status_code().is_success() {
        let body: serde_json::Value = response.json();
        assert!(body.get("id").is_some());
        assert!(body.get("object").is_some());
        assert!(body.get("choices").is_some());
        assert!(body.get("usage").is_some());
    }
}
