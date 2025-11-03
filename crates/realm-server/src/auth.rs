//! Authentication and API Key Management
//!
//! Provides API key-based authentication for multi-tenant WebSocket connections.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// API key with associated tenant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// The API key value (e.g., "sk_live_abc123...")
    pub key: String,

    /// Tenant ID this key belongs to
    pub tenant_id: String,

    /// Human-readable name for this key
    pub name: Option<String>,

    /// Whether this key is active
    pub enabled: bool,

    /// Optional rate limit override (requests per minute)
    pub rate_limit: Option<usize>,

    /// Creation timestamp
    #[serde(default)]
    pub created_at: i64,

    /// Last used timestamp
    #[serde(default)]
    pub last_used: Option<i64>,
}

impl ApiKey {
    /// Create a new API key
    pub fn new(key: impl Into<String>, tenant_id: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            tenant_id: tenant_id.into(),
            name: None,
            enabled: true,
            rate_limit: None,
            created_at: chrono::Utc::now().timestamp(),
            last_used: None,
        }
    }

    /// Set a human-readable name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a custom rate limit
    pub fn with_rate_limit(mut self, rate_limit: usize) -> Self {
        self.rate_limit = Some(rate_limit);
        self
    }

    /// Check if key is valid (enabled)
    pub fn is_valid(&self) -> bool {
        self.enabled
    }
}

/// API key store configuration
#[derive(Debug, Clone)]
pub struct ApiKeyStoreConfig {
    /// Path to API keys file (JSON)
    pub keys_file: Option<PathBuf>,

    /// Default rate limit (requests per minute)
    pub default_rate_limit: usize,

    /// Enable in-memory only mode (no persistence)
    pub in_memory_only: bool,
}

impl Default for ApiKeyStoreConfig {
    fn default() -> Self {
        Self {
            keys_file: None,
            default_rate_limit: 60,
            in_memory_only: false,
        }
    }
}

/// API key store with validation and persistence
pub struct ApiKeyStore {
    /// Configuration
    config: ApiKeyStoreConfig,

    /// In-memory key storage (key -> ApiKey)
    keys: Arc<RwLock<HashMap<String, ApiKey>>>,
}

impl ApiKeyStore {
    /// Create a new API key store
    pub fn new(config: ApiKeyStoreConfig) -> Result<Self> {
        let keys_file_path = config.keys_file.clone();

        let mut store = Self {
            config,
            keys: Arc::new(RwLock::new(HashMap::new())),
        };

        // Load keys from file if configured
        if let Some(keys_file) = keys_file_path {
            if keys_file.exists() {
                store.load_from_file(&keys_file)?;
            } else {
                info!("API keys file not found, starting with empty store");
            }
        }

        Ok(store)
    }

    /// Create a store with in-memory keys (no persistence)
    pub fn in_memory() -> Self {
        Self {
            config: ApiKeyStoreConfig {
                in_memory_only: true,
                ..Default::default()
            },
            keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add an API key to the store
    pub fn add_key(&self, key: ApiKey) -> Result<()> {
        let mut keys = self.keys.write().unwrap();

        if keys.contains_key(&key.key) {
            return Err(anyhow!("API key already exists: {}", key.key));
        }

        info!(
            "Adding API key for tenant: {} (name: {:?})",
            key.tenant_id, key.name
        );

        keys.insert(key.key.clone(), key);

        // Persist to file if configured
        if !self.config.in_memory_only {
            drop(keys); // Release lock before saving
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Remove an API key
    pub fn remove_key(&self, key: &str) -> Result<()> {
        let mut keys = self.keys.write().unwrap();

        keys.remove(key)
            .ok_or_else(|| anyhow!("API key not found: {}", key))?;

        info!("Removed API key: {}", key);

        // Persist to file if configured
        if !self.config.in_memory_only {
            drop(keys); // Release lock before saving
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Validate an API key and return tenant ID
    pub fn validate(&self, key: &str) -> Result<String> {
        let mut keys = self.keys.write().unwrap();

        let api_key = keys
            .get_mut(key)
            .ok_or_else(|| anyhow!("Invalid API key"))?;

        if !api_key.is_valid() {
            warn!("Attempted use of disabled API key: {}", key);
            return Err(anyhow!("API key is disabled"));
        }

        // Update last used timestamp
        api_key.last_used = Some(chrono::Utc::now().timestamp());

        debug!("Validated API key for tenant: {}", api_key.tenant_id);

        Ok(api_key.tenant_id.clone())
    }

    /// Get API key details
    pub fn get_key(&self, key: &str) -> Option<ApiKey> {
        let keys = self.keys.read().unwrap();
        keys.get(key).cloned()
    }

    /// List all API keys for a tenant
    pub fn list_tenant_keys(&self, tenant_id: &str) -> Vec<ApiKey> {
        let keys = self.keys.read().unwrap();
        keys.values()
            .filter(|k| k.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Get all API keys (admin only)
    pub fn list_all_keys(&self) -> Vec<ApiKey> {
        let keys = self.keys.read().unwrap();
        keys.values().cloned().collect()
    }

    /// Disable an API key
    pub fn disable_key(&self, key: &str) -> Result<()> {
        let mut keys = self.keys.write().unwrap();

        let api_key = keys
            .get_mut(key)
            .ok_or_else(|| anyhow!("API key not found: {}", key))?;

        api_key.enabled = false;
        info!("Disabled API key: {}", key);

        // Persist to file if configured
        if !self.config.in_memory_only {
            drop(keys); // Release lock before saving
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Enable an API key
    pub fn enable_key(&self, key: &str) -> Result<()> {
        let mut keys = self.keys.write().unwrap();

        let api_key = keys
            .get_mut(key)
            .ok_or_else(|| anyhow!("API key not found: {}", key))?;

        api_key.enabled = true;
        info!("Enabled API key: {}", key);

        // Persist to file if configured
        if !self.config.in_memory_only {
            drop(keys); // Release lock before saving
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Load keys from JSON file
    fn load_from_file(&mut self, path: &PathBuf) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let keys: Vec<ApiKey> = serde_json::from_str(&content)?;

        let mut store = self.keys.write().unwrap();
        for key in keys {
            store.insert(key.key.clone(), key);
        }

        info!("Loaded {} API keys from {:?}", store.len(), path);

        Ok(())
    }

    /// Save keys to JSON file
    fn save_to_file(&self) -> Result<()> {
        if let Some(ref keys_file) = self.config.keys_file {
            let keys = self.keys.read().unwrap();
            let keys_vec: Vec<&ApiKey> = keys.values().collect();

            let content = serde_json::to_string_pretty(&keys_vec)?;
            std::fs::write(keys_file, content)?;

            debug!("Saved {} API keys to {:?}", keys_vec.len(), keys_file);
        }

        Ok(())
    }

    /// Get default rate limit
    pub fn default_rate_limit(&self) -> usize {
        self.config.default_rate_limit
    }
}

/// Generate a secure API key
pub fn generate_api_key(prefix: &str) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const KEY_LENGTH: usize = 32;

    let mut rng = rand::thread_rng();
    let key: String = (0..KEY_LENGTH)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();

    format!("{}_{}", prefix, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_creation() {
        let key = ApiKey::new("test_key", "tenant1")
            .with_name("Test Key")
            .with_rate_limit(100);

        assert_eq!(key.key, "test_key");
        assert_eq!(key.tenant_id, "tenant1");
        assert_eq!(key.name, Some("Test Key".to_string()));
        assert_eq!(key.rate_limit, Some(100));
        assert!(key.is_valid());
    }

    #[test]
    fn test_api_key_store_validation() {
        let store = ApiKeyStore::in_memory();

        let key = ApiKey::new("sk_test_123", "tenant1");
        store.add_key(key).unwrap();

        // Valid key
        let tenant_id = store.validate("sk_test_123").unwrap();
        assert_eq!(tenant_id, "tenant1");

        // Invalid key
        assert!(store.validate("sk_invalid").is_err());
    }

    #[test]
    fn test_api_key_disable() {
        let store = ApiKeyStore::in_memory();

        let key = ApiKey::new("sk_test_456", "tenant2");
        store.add_key(key).unwrap();

        // Valid initially
        assert!(store.validate("sk_test_456").is_ok());

        // Disable
        store.disable_key("sk_test_456").unwrap();
        assert!(store.validate("sk_test_456").is_err());

        // Re-enable
        store.enable_key("sk_test_456").unwrap();
        assert!(store.validate("sk_test_456").is_ok());
    }

    #[test]
    fn test_generate_api_key() {
        let key1 = generate_api_key("sk_live");
        let key2 = generate_api_key("sk_live");

        assert!(key1.starts_with("sk_live_"));
        assert!(key2.starts_with("sk_live_"));
        assert_ne!(key1, key2); // Should be unique
        assert_eq!(key1.len(), "sk_live_".len() + 32);
    }

    #[test]
    fn test_list_tenant_keys() {
        let store = ApiKeyStore::in_memory();

        store.add_key(ApiKey::new("key1", "tenant1")).unwrap();
        store.add_key(ApiKey::new("key2", "tenant1")).unwrap();
        store.add_key(ApiKey::new("key3", "tenant2")).unwrap();

        let tenant1_keys = store.list_tenant_keys("tenant1");
        assert_eq!(tenant1_keys.len(), 2);

        let tenant2_keys = store.list_tenant_keys("tenant2");
        assert_eq!(tenant2_keys.len(), 1);
    }
}
