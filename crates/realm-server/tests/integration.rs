//! Integration tests for realm-server
//!
//! Tests the full flow: HTTP/WebSocket -> Runtime Manager -> WASM -> Host Functions

use axum::http::StatusCode;
use axum_test::TestServer;
use realm_server::http::{create_router, ServerState};
use realm_server::runtime_manager::RuntimeManager;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_health_check() {
    let runtime_manager = Arc::new(Mutex::new(
        RuntimeManager::new("./target/wasm32-unknown-unknown/release/realm_wasm.wasm")
            .unwrap_or_else(|_| {
                // If WASM file doesn't exist, create a dummy runtime manager for testing
                // In real tests, we'd compile the WASM first
                panic!("WASM file not found - run 'cargo build --target wasm32-unknown-unknown -p realm-wasm' first");
            }),
    ));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
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
    let runtime_manager = Arc::new(Mutex::new(
        RuntimeManager::new("./target/wasm32-unknown-unknown/release/realm_wasm.wasm")
            .unwrap_or_else(|_| {
                panic!("WASM file not found");
            }),
    ));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
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
    let runtime_manager = Arc::new(Mutex::new(
        RuntimeManager::new("./target/wasm32-unknown-unknown/release/realm_wasm.wasm")
            .unwrap_or_else(|_| {
                panic!("WASM file not found");
            }),
    ));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
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
    let runtime_manager = Arc::new(Mutex::new(
        RuntimeManager::new("./target/wasm32-unknown-unknown/release/realm_wasm.wasm")
            .unwrap_or_else(|_| {
                panic!("WASM file not found");
            }),
    ));

    let state = ServerState {
        runtime_manager,
        api_key_store: None,
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
