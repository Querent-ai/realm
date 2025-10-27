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
