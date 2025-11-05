//! Model Orchestrator
//!
//! Orchestrates multiple models for complex workflows like RAG → LLM → Summarizer.
//! This is Realm's core business differentiator - chaining models together.

use crate::runtime_manager::{ModelConfig, RuntimeManager};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Model type/capability
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Text generation/completion
    Completion,
    /// Chat/conversation
    Chat,
    /// Text embeddings
    Embedding,
    /// Summarization
    Summarization,
    /// Classification
    Classification,
    /// Custom model type
    Custom(String),
}

impl ModelType {
    pub fn as_str(&self) -> &str {
        match self {
            ModelType::Completion => "completion",
            ModelType::Chat => "chat",
            ModelType::Embedding => "embedding",
            ModelType::Summarization => "summarization",
            ModelType::Classification => "classification",
            ModelType::Custom(name) => name.as_str(),
        }
    }
}

/// Model specification for the orchestrator
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Unique model identifier
    pub model_id: String,

    /// Model type/capability
    pub model_type: ModelType,

    /// Path to GGUF model file
    pub model_path: PathBuf,

    /// Human-readable name
    pub name: String,

    /// Optional description
    pub description: Option<String>,

    /// Model configuration
    pub config: ModelConfig,
}

impl ModelSpec {
    pub fn new(
        model_id: impl Into<String>,
        model_type: ModelType,
        model_path: impl Into<PathBuf>,
        name: impl Into<String>,
    ) -> Self {
        let model_id = model_id.into();
        let model_path = model_path.into();
        let name = name.into();

        Self {
            model_id: model_id.clone(),
            model_type,
            model_path: model_path.clone(),
            name,
            description: None,
            config: ModelConfig {
                model_path,
                model_id,
                draft_model_path: None,
                draft_model_id: None,
            },
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Pipeline step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Step name/identifier
    pub name: String,

    /// Model ID to use for this step
    pub model_id: String,

    /// Input field name (from previous step's output)
    pub input_field: Option<String>,

    /// Output field name (for next step)
    pub output_field: String,

    /// Whether this step is optional
    #[serde(default)]
    pub optional: bool,
}

/// Pipeline definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Pipeline name/identifier
    pub name: String,

    /// Pipeline steps (executed in order)
    pub steps: Vec<PipelineStep>,

    /// Default input field (for first step)
    pub default_input_field: String,
}

impl Pipeline {
    /// Create a simple linear pipeline
    pub fn linear(name: impl Into<String>, model_ids: Vec<String>) -> Self {
        let name = name.into();
        let steps = model_ids
            .iter()
            .enumerate()
            .map(|(i, model_id)| PipelineStep {
                name: format!("step_{}", i),
                model_id: model_id.clone(),
                input_field: if i == 0 {
                    None
                } else {
                    Some(format!("output_{}", i - 1))
                },
                output_field: format!("output_{}", i),
                optional: false,
            })
            .collect();

        Self {
            name,
            steps,
            default_input_field: "input".to_string(),
        }
    }
}

/// Execution context for a pipeline run
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Input data
    pub input: serde_json::Value,

    /// Intermediate results (step_name -> output)
    pub intermediates: HashMap<String, serde_json::Value>,

    /// Final output
    pub output: Option<serde_json::Value>,
}

impl PipelineContext {
    pub fn new(input: serde_json::Value) -> Self {
        Self {
            input,
            intermediates: HashMap::new(),
            output: None,
        }
    }

    pub fn set_step_output(&mut self, step_name: &str, output: serde_json::Value) {
        self.intermediates.insert(step_name.to_string(), output);
    }

    pub fn get_step_input(&self, step: &PipelineStep) -> Option<&serde_json::Value> {
        if let Some(ref input_field) = step.input_field {
            self.intermediates.get(input_field)
        } else {
            Some(&self.input)
        }
    }
}

/// Model orchestrator for multi-model workflows
pub struct ModelOrchestrator {
    /// Runtime manager for WASM execution
    runtime_manager: Arc<RuntimeManager>,

    /// Registered models (model_id -> ModelSpec)
    models: Arc<Mutex<HashMap<String, ModelSpec>>>,

    /// Models by type (type -> Vec<model_id>)
    models_by_type: Arc<Mutex<HashMap<ModelType, Vec<String>>>>,

    /// Registered pipelines
    pipelines: Arc<Mutex<HashMap<String, Pipeline>>>,

    /// Default model per type (for routing)
    default_models: Arc<Mutex<HashMap<ModelType, String>>>,
}

impl ModelOrchestrator {
    /// Create a new model orchestrator
    pub fn new(runtime_manager: Arc<RuntimeManager>) -> Self {
        Self {
            runtime_manager,
            models: Arc::new(Mutex::new(HashMap::new())),
            models_by_type: Arc::new(Mutex::new(HashMap::new())),
            pipelines: Arc::new(Mutex::new(HashMap::new())),
            default_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a model with the orchestrator
    pub fn register_model(&self, spec: ModelSpec) -> Result<()> {
        let model_id = spec.model_id.clone();
        let model_type = spec.model_type.clone();

        info!(
            "Registering model: {} (type: {:?})",
            model_id,
            model_type.as_str()
        );

        // Validate model file exists
        if !spec.model_path.exists() {
            return Err(anyhow!("Model file does not exist: {:?}", spec.model_path));
        }

        // Store model spec
        let mut models = self.models.lock().unwrap();
        models.insert(model_id.clone(), spec.clone());

        // Index by type
        let mut models_by_type = self.models_by_type.lock().unwrap();
        models_by_type
            .entry(model_type.clone())
            .or_default()
            .push(model_id.clone());

        // Set as default if first model of this type
        let mut default_models = self.default_models.lock().unwrap();
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = default_models.entry(model_type) {
            e.insert(model_id.clone());
            info!("Set {} as default model for type", model_id);
        }

        Ok(())
    }

    /// Register a pipeline
    pub fn register_pipeline(&self, pipeline: Pipeline) -> Result<()> {
        let name = pipeline.name.clone();

        // Validate pipeline: check all model IDs exist
        let models = self.models.lock().unwrap();
        for step in &pipeline.steps {
            if !models.contains_key(&step.model_id) {
                return Err(anyhow!(
                    "Pipeline '{}' references unknown model: {}",
                    name,
                    step.model_id
                ));
            }
        }

        let mut pipelines = self.pipelines.lock().unwrap();
        pipelines.insert(name.clone(), pipeline);

        info!("Registered pipeline: {}", name);
        Ok(())
    }

    /// Get a model by ID
    pub fn get_model(&self, model_id: &str) -> Option<ModelSpec> {
        let models = self.models.lock().unwrap();
        models.get(model_id).cloned()
    }

    /// Get models by type
    pub fn get_models_by_type(&self, model_type: &ModelType) -> Vec<ModelSpec> {
        let models = self.models.lock().unwrap();
        let models_by_type = self.models_by_type.lock().unwrap();

        models_by_type
            .get(model_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| models.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get default model for a type
    pub fn get_default_model(&self, model_type: &ModelType) -> Option<String> {
        let default_models = self.default_models.lock().unwrap();
        default_models.get(model_type).cloned()
    }

    /// Set default model for a type
    pub fn set_default_model(&self, model_type: ModelType, model_id: String) -> Result<()> {
        // Validate model exists
        let models = self.models.lock().unwrap();
        if !models.contains_key(&model_id) {
            return Err(anyhow!("Model not found: {}", model_id));
        }

        let mut default_models = self.default_models.lock().unwrap();
        default_models.insert(model_type, model_id.clone());

        info!("Set {} as default model", model_id);
        Ok(())
    }

    /// Route a request to the appropriate model based on type
    pub fn route_request(&self, model_type: &ModelType) -> Result<String> {
        let default_model = self
            .get_default_model(model_type)
            .ok_or_else(|| anyhow!("No default model for type: {:?}", model_type))?;

        debug!(
            "Routed request to model: {} (type: {:?})",
            default_model, model_type
        );
        Ok(default_model)
    }

    /// Execute a pipeline
    pub async fn execute_pipeline(
        &self,
        pipeline_name: &str,
        tenant_id: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        info!(
            "Executing pipeline '{}' for tenant '{}'",
            pipeline_name, tenant_id
        );

        // Get pipeline
        let pipelines = self.pipelines.lock().unwrap();
        let pipeline = pipelines
            .get(pipeline_name)
            .ok_or_else(|| anyhow!("Pipeline not found: {}", pipeline_name))?
            .clone();
        drop(pipelines);

        // Ensure tenant runtime exists
        self.runtime_manager.get_or_create_runtime(tenant_id)?;

        // Create execution context
        let mut context = PipelineContext::new(input);

        // Execute steps sequentially
        for step in &pipeline.steps {
            debug!(
                "Executing pipeline step: {} (model: {})",
                step.name, step.model_id
            );

            // Get input for this step
            let step_input = context
                .get_step_input(step)
                .ok_or_else(|| anyhow!("Missing input for step: {}", step.name))?
                .clone();

            // Extract prompt from input
            let prompt = if step_input.is_string() {
                step_input.as_str().unwrap().to_string()
            } else if let Some(prompt) = step_input.get("prompt") {
                prompt.as_str().unwrap().to_string()
            } else if let Some(text) = step_input.get("text") {
                text.as_str().unwrap().to_string()
            } else {
                return Err(anyhow!("Could not extract prompt from step input"));
            };

            // Execute model inference
            let output_text = self
                .runtime_manager
                .generate(tenant_id, prompt.clone())
                .context(format!("Pipeline step '{}' failed", step.name))?;

            // Store step output
            let step_output = serde_json::json!({
                "text": output_text,
                "model_id": step.model_id,
                "step": step.name,
            });

            context.set_step_output(&step.output_field, step_output.clone());

            // If this is the last step, set as final output
            if step.output_field == pipeline.steps.last().unwrap().output_field {
                context.output = Some(step_output);
            }
        }

        context
            .output
            .ok_or_else(|| anyhow!("Pipeline execution completed but no output was produced"))
    }

    /// Execute a single model inference
    pub async fn execute_model(
        &self,
        model_id: &str,
        tenant_id: &str,
        prompt: String,
    ) -> Result<String> {
        debug!("Executing model '{}' for tenant '{}'", model_id, tenant_id);

        // Validate model exists
        let _spec = self
            .get_model(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;

        // Ensure tenant runtime exists
        self.runtime_manager.get_or_create_runtime(tenant_id)?;

        // Execute inference
        self.runtime_manager
            .generate(tenant_id, prompt)
            .context(format!("Model '{}' execution failed", model_id))
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<ModelSpec> {
        let models = self.models.lock().unwrap();
        models.values().cloned().collect()
    }

    /// List all registered pipelines
    pub fn list_pipelines(&self) -> Vec<String> {
        let pipelines = self.pipelines.lock().unwrap();
        pipelines.keys().cloned().collect()
    }

    /// Get pipeline definition
    pub fn get_pipeline(&self, name: &str) -> Option<Pipeline> {
        let pipelines = self.pipelines.lock().unwrap();
        pipelines.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_spec_creation() {
        let spec = ModelSpec::new(
            "llama-2-7b",
            ModelType::Completion,
            PathBuf::from("/models/llama-2-7b.gguf"),
            "Llama 2 7B",
        )
        .with_description("7B parameter completion model");

        assert_eq!(spec.model_id, "llama-2-7b");
        assert_eq!(spec.model_type, ModelType::Completion);
        assert_eq!(spec.name, "Llama 2 7B");
    }

    #[test]
    fn test_pipeline_linear_creation() {
        let pipeline = Pipeline::linear(
            "rag-pipeline",
            vec!["embedding-model".to_string(), "llm-model".to_string()],
        );

        assert_eq!(pipeline.name, "rag-pipeline");
        assert_eq!(pipeline.steps.len(), 2);
        assert_eq!(pipeline.steps[0].model_id, "embedding-model");
        assert_eq!(pipeline.steps[1].model_id, "llm-model");
    }

    #[test]
    fn test_pipeline_context() {
        let mut context = PipelineContext::new(serde_json::json!({"text": "Hello"}));

        context.set_step_output("step1", serde_json::json!({"output": "World"}));
        assert!(context.intermediates.contains_key("step1"));

        let step = PipelineStep {
            name: "step1".to_string(),
            model_id: "model1".to_string(),
            input_field: None,
            output_field: "output".to_string(),
            optional: false,
        };

        let input = context.get_step_input(&step);
        assert!(input.is_some());
    }
}
