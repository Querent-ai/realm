use thiserror::Error;

/// Core error types for wasm-chord
#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, Error>;
