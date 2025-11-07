.PHONY: help build test lint clean install fmt check bench docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all crates in release mode
	@echo "Building all crates..."
	cargo build --release
	@echo "Building WASM module..."
	cd crates/realm-wasm && wasm-pack build --target web

build-dev: ## Build all crates in debug mode
	cargo build

build-cuda: ## Build with CUDA support
	cargo build --release --features cuda

build-metal: ## Build with Metal support (macOS only)
	cargo build --release --features metal

test: ## Run all tests
	cargo test --workspace

test-cuda: ## Run tests with CUDA
	cargo test --workspace --features cuda

test-unit: ## Run unit tests only
	cargo test --lib --workspace

test-integration: ## Run integration tests
	cargo test --test '*'

lint: ## Run clippy lints
	cargo clippy --workspace --all-targets -- -D warnings

lint-fix: ## Run clippy with automatic fixes
	cargo clippy --workspace --all-targets --fix --allow-dirty -- -D warnings

fmt: ## Format code with rustfmt
	cargo fmt --all

fmt-check: ## Check code formatting
	cargo fmt --all -- --check

check: fmt-check lint test ## Run all checks (fmt, lint, test)

bench: ## Run benchmarks
	cargo bench --workspace

docs: ## Build documentation
	cargo doc --no-deps --workspace --open

docs-all: ## Build documentation with dependencies
	cargo doc --workspace --open

clean: ## Clean build artifacts
	cargo clean
	rm -rf crates/realm-wasm/pkg
	rm -rf target

install: ## Install realm CLI
	cargo install --path cli

# Development helpers
watch: ## Watch for changes and run tests
	cargo watch -x test

watch-build: ## Watch for changes and build
	cargo watch -x build

# CI targets
ci-build: ## CI build target
	cargo build --workspace --all-targets
	cd crates/realm-wasm && wasm-pack build --target web

ci-test: ## CI test target
	cargo test --workspace --all-targets

ci-lint: ## CI lint target
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

ci: ci-build ci-test ci-lint ## Run all CI checks

# E2E test targets
e2e: e2e-full ## Run e2e tests (starts server, runs tests, cleans up)

e2e-setup: ## Setup e2e test environment (build binaries, install deps)
	@echo "📦 Setting up E2E test environment..."
	@if [ ! -f "target/release/realm" ]; then \
		echo "  Building realm server..."; \
		cargo build --release --bin realm; \
	fi
	@if [ ! -f "crates/realm-wasm/pkg/realm_wasm_bg.wasm" ] && [ ! -f "wasm-pkg/realm_wasm_bg.wasm" ]; then \
		echo "  Building WASM module..."; \
		cd crates/realm-wasm && wasm-pack build --target web && cd ../..; \
	fi
	@if [ ! -d "e2e/node_modules" ]; then \
		echo "  Installing e2e dependencies..."; \
		cd e2e && npm install && cd ..; \
	fi
	@echo "✓ E2E setup complete"

e2e-run: ## Run e2e tests (assumes server is running)
	@echo "🧪 Running E2E tests..."
	@cd e2e && REALM_SERVER_URL=http://localhost:3000 npm run test:all

e2e-server: ## Start server for e2e tests (runs in background)
	@echo "🚀 Starting Realm server for E2E tests..."
	@if [ ! -f "/tmp/realm-e2e-server.pid" ] || ! kill -0 $$(cat /tmp/realm-e2e-server.pid) 2>/dev/null; then \
		WASM_FILE=""; \
		if [ -f "crates/realm-wasm/pkg/realm_wasm_bg.wasm" ]; then \
			WASM_FILE="crates/realm-wasm/pkg/realm_wasm_bg.wasm"; \
		elif [ -f "wasm-pkg/realm_wasm_bg.wasm" ]; then \
			WASM_FILE="wasm-pkg/realm_wasm_bg.wasm"; \
		fi; \
		MODEL_FILE=""; \
		if [ -f "models/tinyllama-1.1b.Q4_K_M.gguf" ]; then \
			MODEL_FILE="models/tinyllama-1.1b.Q4_K_M.gguf"; \
		elif [ -f "models/llama-2-7b-chat-q4_k_m.gguf" ]; then \
			MODEL_FILE="models/llama-2-7b-chat-q4_k_m.gguf"; \
		fi; \
		if [ -n "$$WASM_FILE" ] && [ -n "$$MODEL_FILE" ]; then \
			echo "  Starting with WASM: $$WASM_FILE and model: $$MODEL_FILE"; \
			./target/release/realm serve \
				--wasm "$$WASM_FILE" \
				--model "$$MODEL_FILE" \
				--host 127.0.0.1 \
				--port 3000 \
				--http \
				--http-port 3000 \
				> /tmp/realm-e2e-server.log 2>&1 & \
			echo $$! > /tmp/realm-e2e-server.pid; \
		else \
			echo "  ⚠️  Warning: WASM or model file not found, server may not work correctly"; \
			echo "     WASM: $$WASM_FILE"; \
			echo "     Model: $$MODEL_FILE"; \
			./target/release/realm serve \
				--host 127.0.0.1 \
				--port 3000 \
				--http \
				--http-port 3000 \
				> /tmp/realm-e2e-server.log 2>&1 & \
			echo $$! > /tmp/realm-e2e-server.pid; \
		fi; \
		echo "  Waiting for server to be ready..."; \
		for i in $$(seq 1 30); do \
			if curl -s http://localhost:3000/health > /dev/null 2>&1; then \
				echo "  ✓ Server is ready (PID: $$(cat /tmp/realm-e2e-server.pid))"; \
				exit 0; \
			fi; \
			if [ $$i -eq 30 ]; then \
				echo "  ❌ Server failed to start after 30 seconds"; \
				echo "  Logs:"; \
				tail -20 /tmp/realm-e2e-server.log; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
	else \
		echo "  Server already running (PID: $$(cat /tmp/realm-e2e-server.pid))"; \
	fi

e2e-stop: ## Stop e2e test server
	@if [ -f "/tmp/realm-e2e-server.pid" ]; then \
		SERVER_PID=$$(cat /tmp/realm-e2e-server.pid); \
		if kill -0 $$SERVER_PID 2>/dev/null; then \
			echo "🛑 Stopping E2E server (PID: $$SERVER_PID)..."; \
			kill $$SERVER_PID 2>/dev/null || true; \
			wait $$SERVER_PID 2>/dev/null || true; \
		fi; \
		rm -f /tmp/realm-e2e-server.pid; \
		echo "✓ Server stopped"; \
	else \
		echo "  No server PID file found"; \
	fi

e2e-cleanup: e2e-stop ## Cleanup e2e test environment

e2e-full: ## Full e2e test run (setup, start server, run tests, cleanup)
	@echo "🧪 Running full E2E test suite..."
	@$(MAKE) e2e-setup
	@$(MAKE) e2e-server || (echo "❌ Failed to start server"; exit 1)
	@trap '$(MAKE) e2e-cleanup' EXIT INT TERM; \
		cd e2e && REALM_SERVER_URL=http://localhost:3000 npm run test:all; \
		TEST_RESULT=$$?; \
		$(MAKE) e2e-cleanup; \
		exit $$TEST_RESULT
