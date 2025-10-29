# Contributing to Realm

Thank you for your interest in contributing to Realm! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and considerate. We're all here to build something great together.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Git
- For WASM development: `wasm-pack` and `wasm32-unknown-unknown` target
- For GPU development: CUDA 11.8+ or Metal SDK

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/realm.git
cd realm

# Add upstream remote
git remote add upstream https://github.com/querent-ai/realm.git

# Install Rust if you haven't already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build and test
cargo build
cargo test
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, concise code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Format code
cargo fmt

# Or use make
make check
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "feat: add support for X"
# or
git commit -m "fix: resolve issue with Y"
# or
git commit -m "docs: update README with Z"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Build process or auxiliary tool changes

### 5. Push and Create a Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub with:
- Clear description of the changes
- Link to any related issues
- Screenshots if applicable (for UI changes)

## Code Style

### Rust

- Run `cargo fmt` before committing
- Run `cargo clippy` and address all warnings
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use meaningful variable and function names
- Add doc comments for public APIs

```rust
/// Computes the attention weights for a given query and key.
///
/// # Arguments
///
/// * `query` - The query tensor (batch_size, seq_len, d_model)
/// * `key` - The key tensor (batch_size, kv_len, d_model)
///
/// # Returns
///
/// Attention weights tensor (batch_size, seq_len, kv_len)
pub fn compute_attention(query: &Tensor, key: &Tensor) -> Result<Tensor> {
    // Implementation
}
```

### Documentation

- Use clear, concise language
- Provide examples where helpful
- Keep documentation up to date with code changes

## Testing

### Unit Tests

Add unit tests in the same file as your code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_computation() {
        let query = Tensor::zeros(&[1, 10, 512]);
        let key = Tensor::zeros(&[1, 10, 512]);

        let result = compute_attention(&query, &key).unwrap();

        assert_eq!(result.shape(), &[1, 10, 10]);
    }
}
```

### Integration Tests

Add integration tests in `tests/` directory:

```rust
// tests/integration_test.rs
use realm_runtime::*;

#[test]
fn test_end_to_end_inference() {
    // Test implementation
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p realm-core

# Specific test
cargo test test_attention_computation

# With logging
cargo test -- --nocapture
```

## Pull Request Process

1. **Update Documentation**: Ensure README, doc comments, and guides are updated
2. **Add Tests**: New functionality should have corresponding tests
3. **Pass CI**: All CI checks must pass
4. **Request Review**: Tag relevant maintainers for review
5. **Address Feedback**: Respond to review comments promptly
6. **Squash Commits**: Maintainers may ask you to squash commits before merging

## Areas for Contribution

### High Priority

- 🔴 **Performance**: Optimize kernels, reduce latency
- 🔴 **Testing**: Add unit and integration tests
- 🔴 **Documentation**: Improve guides and examples
- 🔴 **Bug Fixes**: Fix reported issues

### Medium Priority

- 🟡 **Features**: Implement planned features (see README status)
- 🟡 **Examples**: Add more usage examples
- 🟡 **Tooling**: Improve CLI, SDKs

### Good First Issues

Look for issues labeled `good-first-issue` on GitHub. These are great entry points for new contributors.

## Project Structure

```
realm/
├── crates/
│   ├── realm-core/          # Core primitives (GGUF, tokenization)
│   ├── realm-models/        # Transformer models
│   ├── realm-compute-cpu/   # CPU compute backends
│   ├── realm-compute-gpu/   # GPU compute backends
│   ├── realm-runtime/       # Runtime host (Memory64, Wasmtime)
│   └── realm-wasm/          # WASM module
├── examples/                # Usage examples
├── docs/                    # Documentation
└── .github/                 # CI/CD workflows
```

## Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Steps to reproduce
3. **Expected**: What you expected to happen
4. **Actual**: What actually happened
5. **Environment**: OS, Rust version, GPU (if applicable)
6. **Logs**: Relevant error messages or logs

Use the bug report template on GitHub.

## Requesting Features

When requesting features:

1. **Use Case**: Describe the use case
2. **Proposed Solution**: Your idea for how it should work
3. **Alternatives**: Alternative solutions you've considered
4. **Additional Context**: Any other relevant information

Use the feature request template on GitHub.

## Questions?

- 💬 [GitHub Discussions](https://github.com/querent-ai/realm/discussions) - Ask questions
- 💭 [Discord](https://discord.gg/realm) - Chat with the community
- 📧 Email: [opensource@realm.ai](mailto:opensource@realm.ai)

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
