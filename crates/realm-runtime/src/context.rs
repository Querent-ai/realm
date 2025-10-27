use std::collections::HashMap;
use realm_core::error::Result;

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_memory_bytes: usize,
    pub deterministic: bool,
    pub gpu_enabled: bool,
    pub num_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 4_294_967_295, // ~4 GB (max for 32-bit usize)
            deterministic: false,
            gpu_enabled: true,
            num_threads: 0, // auto-detect
        }
    }
}

impl RuntimeConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            realm_core::error::Error::ParseError(format!("Invalid config JSON: {}", e))
        })
    }
}

impl serde::Serialize for RuntimeConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RuntimeConfig", 4)?;
        state.serialize_field("max_memory_bytes", &self.max_memory_bytes)?;
        state.serialize_field("deterministic", &self.deterministic)?;
        state.serialize_field("gpu_enabled", &self.gpu_enabled)?;
        state.serialize_field("num_threads", &self.num_threads)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for RuntimeConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct ConfigHelper {
            max_memory_bytes: Option<usize>,
            deterministic: Option<bool>,
            gpu_enabled: Option<bool>,
            num_threads: Option<usize>,
        }

        let helper = ConfigHelper::deserialize(deserializer)?;
        Ok(Self {
            max_memory_bytes: helper.max_memory_bytes.unwrap_or(4_294_967_295),
            deterministic: helper.deterministic.unwrap_or(false),
            gpu_enabled: helper.gpu_enabled.unwrap_or(true),
            num_threads: helper.num_threads.unwrap_or(0),
        })
    }
}

/// Global runtime context
pub struct RuntimeContext {
    pub config: RuntimeConfig,
    pub models: HashMap<u32, ModelHandle>,
    next_id: u32,
}

impl RuntimeContext {
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config, models: HashMap::new(), next_id: 1 }
    }

    pub fn allocate_model_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn register_model(&mut self, model: ModelHandle) -> u32 {
        let id = self.allocate_model_id();
        self.models.insert(id, model);
        id
    }

    pub fn get_model(&self, id: u32) -> Option<&ModelHandle> {
        self.models.get(&id)
    }

    pub fn remove_model(&mut self, id: u32) -> Option<ModelHandle> {
        self.models.remove(&id)
    }
}

/// Handle to a loaded model
pub struct ModelHandle {
    pub name: String,
    pub meta: realm_core::formats::gguf::ModelMeta,
    // Model weights would be stored here
    // For now, just metadata
}
