//! Load tests for realm-server HTTP/SSE endpoints
//!
//! Tests concurrent request handling, throughput, and latency under load.
//! These tests help verify production readiness.
//!
//! Note: These tests require a running server. Start the server first:
//!   cargo run --release -p realm-cli -- serve --http --http-port 8081 --wasm <path> --model <path>

use reqwest::Client;
use serde_json::json;
use std::time::{Duration, Instant};

/// Base URL for the test server
const BASE_URL: &str = "http://127.0.0.1:8081";

/// Helper to check if server is running
async fn check_server_available() -> bool {
    let client = Client::new();
    client
        .get(format!("{}/health", BASE_URL))
        .send()
        .await
        .is_ok()
}

#[tokio::test]
#[ignore] // Requires running server - run with: cargo test -- --ignored
async fn test_concurrent_health_checks() {
    if !check_server_available().await {
        eprintln!(
            "⚠️  Server not available at {}, skipping load test",
            BASE_URL
        );
        eprintln!("   Start server first: cargo run --release -p realm-cli -- serve --http --http-port 8081");
        return;
    }

    let client = Client::new();
    let num_requests = 100;
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..num_requests {
        let client_clone = client.clone();
        handles.push(tokio::spawn(async move {
            let start_req = Instant::now();
            let result = client_clone
                .get(format!("{}/health", BASE_URL))
                .send()
                .await;
            let latency = start_req.elapsed();
            (result, latency)
        }));
    }

    let mut success_count = 0;
    let mut error_count = 0;
    let mut latencies = Vec::new();

    for handle in handles {
        match handle.await {
            Ok((Ok(response), latency)) if response.status().is_success() => {
                success_count += 1;
                latencies.push(latency);
            }
            _ => error_count += 1,
        }
    }

    let duration = start.elapsed();
    let rps = num_requests as f64 / duration.as_secs_f64();

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95) / 100];
    let p99 = latencies[(latencies.len() * 99) / 100];

    println!("\n📊 Concurrent Health Checks Results:");
    println!("   Requests: {}", num_requests);
    println!("   Successful: {}", success_count);
    println!("   Errors: {}", error_count);
    println!("   Duration: {:?}", duration);
    println!("   Throughput: {:.2} req/s", rps);
    println!("   P50 Latency: {:?}", p50);
    println!("   P95 Latency: {:?}", p95);
    println!("   P99 Latency: {:?}", p99);

    assert_eq!(
        success_count, num_requests,
        "All health checks should succeed"
    );
    assert!(rps > 50.0, "Should handle at least 50 req/s");
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_concurrent_chat_completions() {
    if !check_server_available().await {
        eprintln!("⚠️  Server not available, skipping test");
        return;
    }

    let client = Client::new();
    let num_requests = 50;
    let start = Instant::now();

    let request = json!({
        "model": "default",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "stream": false,
        "max_tokens": 10
    });

    let mut handles = Vec::new();
    for _ in 0..num_requests {
        let client_clone = client.clone();
        let request_clone = request.clone();
        handles.push(tokio::spawn(async move {
            let start_req = Instant::now();
            let result = client_clone
                .post(format!("{}/v1/chat/completions", BASE_URL))
                .json(&request_clone)
                .send()
                .await;
            let latency = start_req.elapsed();
            (result, latency)
        }));
    }

    let mut success_count = 0;
    let mut error_count = 0;
    let mut latencies = Vec::new();

    for handle in handles {
        match handle.await {
            Ok((Ok(response), latency)) if response.status().is_success() => {
                success_count += 1;
                latencies.push(latency);
            }
            Ok((Ok(response), _)) => {
                error_count += 1;
                eprintln!(
                    "Error response {}: {:?}",
                    response.status(),
                    response.text().await
                );
            }
            Ok((Err(e), _)) => {
                error_count += 1;
                eprintln!("Request failed: {}", e);
            }
            Err(e) => {
                error_count += 1;
                eprintln!("Task failed: {}", e);
            }
        }
    }

    let duration = start.elapsed();
    let rps = num_requests as f64 / duration.as_secs_f64();

    latencies.sort();
    let avg_latency = if !latencies.is_empty() {
        latencies.iter().sum::<Duration>() / latencies.len() as u32
    } else {
        Duration::ZERO
    };
    let p50 = latencies
        .get(latencies.len() / 2)
        .copied()
        .unwrap_or(Duration::ZERO);
    let p95 = latencies
        .get((latencies.len() * 95) / 100)
        .copied()
        .unwrap_or(Duration::ZERO);
    let p99 = latencies
        .get((latencies.len() * 99) / 100)
        .copied()
        .unwrap_or(Duration::ZERO);

    println!("\n📊 Concurrent Chat Completions Results:");
    println!("   Requests: {}", num_requests);
    println!("   Successful: {}", success_count);
    println!("   Errors: {}", error_count);
    println!("   Duration: {:?}", duration);
    println!("   Throughput: {:.2} req/s", rps);
    println!("   Avg Latency: {:?}", avg_latency);
    println!("   P50 Latency: {:?}", p50);
    println!("   P95 Latency: {:?}", p95);
    println!("   P99 Latency: {:?}", p99);

    // Allow some failures under load, but most should succeed
    assert!(
        success_count as f64 / num_requests as f64 > 0.8,
        "At least 80% of requests should succeed"
    );
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_sustained_load() {
    if !check_server_available().await {
        eprintln!("⚠️  Server not available, skipping test");
        return;
    }

    let client = Client::new();
    let duration = Duration::from_secs(10);
    let target_rps = 10;
    let interval = Duration::from_millis(1000 / target_rps as u64);

    let start = Instant::now();
    let mut request_count = 0;
    let mut handles = Vec::new();

    while start.elapsed() < duration {
        let client_clone = client.clone();
        handles.push(tokio::spawn(async move {
            client_clone
                .get(format!("{}/health", BASE_URL))
                .send()
                .await
        }));

        request_count += 1;
        tokio::time::sleep(interval).await;
    }

    // Wait for all requests to complete
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status().is_success() {
                success_count += 1;
            }
        }
    }

    let actual_duration = start.elapsed();
    let actual_rps = request_count as f64 / actual_duration.as_secs_f64();

    println!("\n📊 Sustained Load Results:");
    println!("   Duration: {:?}", actual_duration);
    println!("   Requests: {}", request_count);
    println!("   Successful: {}", success_count);
    println!("   Actual RPS: {:.2}", actual_rps);
    println!("   Target RPS: {}", target_rps);

    assert!(
        actual_rps >= target_rps as f64 * 0.8,
        "Should maintain at least 80% of target RPS"
    );
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_streaming_throughput() {
    if !check_server_available().await {
        eprintln!("⚠️  Server not available, skipping test");
        return;
    }

    let client = Client::new();
    let num_streams = 10;
    let start = Instant::now();

    let request = json!({
        "model": "default",
        "messages": [
            {"role": "user", "content": "Count to 10"}
        ],
        "stream": true,
        "max_tokens": 50
    });

    let mut handles = Vec::new();
    for _ in 0..num_streams {
        let client_clone = client.clone();
        let request_clone = request.clone();
        handles.push(tokio::spawn(async move {
            client_clone
                .post(format!("{}/v1/chat/completions", BASE_URL))
                .json(&request_clone)
                .send()
                .await
        }));
    }

    let mut success_count = 0;
    let mut error_count = 0;
    let mut total_chunks = 0;

    for handle in handles {
        match handle.await {
            Ok(Ok(response)) if response.status().is_success() => {
                success_count += 1;
                if let Ok(text) = response.text().await {
                    total_chunks += text.matches("data:").count();
                }
            }
            _ => error_count += 1,
        }
    }

    let duration = start.elapsed();
    let chunks_per_second = total_chunks as f64 / duration.as_secs_f64();

    println!("\n📊 Streaming Throughput Results:");
    println!("   Streams: {}", num_streams);
    println!("   Successful: {}", success_count);
    println!("   Errors: {}", error_count);
    println!("   Total Chunks: {}", total_chunks);
    println!("   Duration: {:?}", duration);
    println!("   Chunks/sec: {:.2}", chunks_per_second);

    assert!(
        success_count as f64 / num_streams as f64 > 0.8,
        "At least 80% of streams should succeed"
    );
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_connection_pool_exhaustion() {
    if !check_server_available().await {
        eprintln!("⚠️  Server not available, skipping test");
        return;
    }

    let client = Client::new();
    let num_connections = 200;
    let start = Instant::now();

    let mut handles = Vec::new();
    for i in 0..num_connections {
        let client_clone = client.clone();
        handles.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i % 10)).await;
            client_clone
                .get(format!("{}/health", BASE_URL))
                .send()
                .await
        }));
    }

    let mut success_count = 0;
    let mut error_count = 0;

    for handle in handles {
        match handle.await {
            Ok(Ok(response)) if response.status().is_success() => success_count += 1,
            _ => error_count += 1,
        }
    }

    let duration = start.elapsed();

    println!("\n📊 Connection Pool Exhaustion Test:");
    println!("   Connections: {}", num_connections);
    println!("   Successful: {}", success_count);
    println!("   Errors: {}", error_count);
    println!("   Duration: {:?}", duration);
    println!(
        "   Success Rate: {:.1}%",
        (success_count as f64 / num_connections as f64) * 100.0
    );

    assert!(
        success_count as f64 / num_connections as f64 > 0.9,
        "Should handle at least 90% of connections"
    );
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_mixed_workload() {
    if !check_server_available().await {
        eprintln!("⚠️  Server not available, skipping test");
        return;
    }

    let client = Client::new();
    let num_health = 30;
    let num_chat = 20;
    let num_stream = 10;
    let total = num_health + num_chat + num_stream;

    let start = Instant::now();
    let mut handles = Vec::new();

    // Health checks
    for _ in 0..num_health {
        let client_clone = client.clone();
        handles.push(tokio::spawn(async move {
            let result = client_clone
                .get(format!("{}/health", BASE_URL))
                .send()
                .await;
            ("health", result)
        }));
    }

    // Chat completions
    let chat_request = json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
        "max_tokens": 5
    });
    for _ in 0..num_chat {
        let client_clone = client.clone();
        let request_clone = chat_request.clone();
        handles.push(tokio::spawn(async move {
            let result = client_clone
                .post(format!("{}/v1/chat/completions", BASE_URL))
                .json(&request_clone)
                .send()
                .await;
            ("chat", result)
        }));
    }

    // Streaming
    let stream_request = json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Count"}],
        "stream": true,
        "max_tokens": 10
    });
    for _ in 0..num_stream {
        let client_clone = client.clone();
        let request_clone = stream_request.clone();
        handles.push(tokio::spawn(async move {
            let result = client_clone
                .post(format!("{}/v1/chat/completions", BASE_URL))
                .json(&request_clone)
                .send()
                .await;
            ("stream", result)
        }));
    }

    let mut health_success = 0;
    let mut chat_success = 0;
    let mut stream_success = 0;
    let mut total_errors = 0;

    for handle in handles {
        match handle.await {
            Ok(("health", Ok(response))) if response.status().is_success() => health_success += 1,
            Ok(("chat", Ok(response))) if response.status().is_success() => chat_success += 1,
            Ok(("stream", Ok(response))) if response.status().is_success() => stream_success += 1,
            Ok((endpoint, Ok(response))) => {
                total_errors += 1;
                eprintln!("{} endpoint returned {}", endpoint, response.status());
            }
            Ok((endpoint, Err(e))) => {
                total_errors += 1;
                eprintln!("{} endpoint failed: {}", endpoint, e);
            }
            Err(e) => {
                total_errors += 1;
                eprintln!("Task failed: {}", e);
            }
        }
    }

    let duration = start.elapsed();
    let total_success = health_success + chat_success + stream_success;

    println!("\n📊 Mixed Workload Results:");
    println!("   Total Requests: {}", total);
    println!("   Health Checks: {}/{}", health_success, num_health);
    println!("   Chat Completions: {}/{}", chat_success, num_chat);
    println!("   Streaming: {}/{}", stream_success, num_stream);
    println!("   Total Success: {}", total_success);
    println!("   Total Errors: {}", total_errors);
    println!("   Duration: {:?}", duration);
    println!(
        "   Success Rate: {:.1}%",
        (total_success as f64 / total as f64) * 100.0
    );

    assert!(
        total_success as f64 / total as f64 > 0.8,
        "At least 80% of mixed workload should succeed"
    );
}
