//! Pipeline DSL
//!
//! Declarative pipeline definition using YAML/JSON for multi-model orchestration.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::orchestrator::{ModelType, Pipeline, PipelineStep};

/// Pipeline definition in DSL format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDef {
    /// Pipeline ID
    pub id: String,

    /// Display name
    pub name: String,

    /// Description
    pub description: String,

    /// Pipeline steps
    pub steps: Vec<StepDef>,

    /// Input schema (optional)
    #[serde(default)]
    pub inputs: HashMap<String, InputDef>,

    /// Output mapping (optional)
    #[serde(default)]
    pub outputs: HashMap<String, String>,
}

/// Step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDef {
    /// Step ID
    pub id: String,

    /// Display name
    pub name: String,

    /// Model specification (ID or type)
    #[serde(flatten)]
    pub model: ModelSpec,

    /// Input mapping
    pub input: InputMapping,

    /// Output field name
    #[serde(default = "default_output")]
    pub output: String,
}

fn default_output() -> String {
    "result".to_string()
}

/// Model specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSpec {
    /// Direct model ID
    Id { model: String },

    /// Model type (use default model)
    Type { model_type: String },
}

/// Input mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMapping {
    /// Reference to previous step output
    Reference(String),

    /// Template string with placeholders
    Template { template: String },

    /// Static value
    Static { value: String },
}

/// Input definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputDef {
    /// Input type
    #[serde(rename = "type")]
    pub input_type: String,

    /// Description
    pub description: String,

    /// Required flag
    #[serde(default)]
    pub required: bool,

    /// Default value
    #[serde(default)]
    pub default: Option<String>,
}

impl PipelineDef {
    /// Load pipeline from YAML file
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read pipeline definition file")?;
        serde_yaml::from_str(&content).context("Failed to parse pipeline YAML")
    }

    /// Load pipeline from JSON file
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read pipeline definition file")?;
        serde_json::from_str(&content).context("Failed to parse pipeline JSON")
    }

    /// Load pipeline from YAML string
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).context("Failed to parse pipeline YAML")
    }

    /// Load pipeline from JSON string
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("Failed to parse pipeline JSON")
    }

    /// Convert to orchestrator Pipeline
    pub fn to_pipeline(&self) -> Result<Pipeline> {
        let steps = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| self.convert_step(step, i))
            .collect::<Result<Vec<_>>>()?;

        Ok(Pipeline {
            name: self.id.clone(),
            steps,
            default_input_field: "input".to_string(),
        })
    }

    /// Convert step definition to pipeline step
    fn convert_step(&self, step: &StepDef, index: usize) -> Result<PipelineStep> {
        // Determine model_id
        let model_id = match &step.model {
            ModelSpec::Id { model } => model.clone(),
            ModelSpec::Type { model_type } => {
                // For type-based routing, we use a special syntax
                format!("@type:{}", model_type)
            }
        };

        // Convert input mapping to input field
        let input_field = match &step.input {
            InputMapping::Reference(key) => {
                if key == "input" && index == 0 {
                    None // First step uses default input
                } else {
                    Some(key.clone())
                }
            }
            InputMapping::Template { template } => {
                // For templates, we'll use a special syntax
                Some(format!("@template:{}", template))
            }
            InputMapping::Static { value } => {
                // For static values, we use a special syntax
                Some(format!("@static:{}", value))
            }
        };

        Ok(PipelineStep {
            name: step.name.clone(),
            model_id,
            input_field,
            output_field: step.output.clone(),
            optional: false,
        })
    }

    /// Validate pipeline definition
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate step IDs
        let mut seen_ids = std::collections::HashSet::new();
        for step in &self.steps {
            if !seen_ids.insert(&step.id) {
                return Err(anyhow!("Duplicate step ID: {}", step.id));
            }
        }

        // Validate input references
        for step in &self.steps {
            if let InputMapping::Reference(ref_key) = &step.input {
                // Check if referenced step exists
                if ref_key != "input" && !self.steps.iter().any(|s| s.output == *ref_key) {
                    return Err(anyhow!(
                        "Step '{}' references unknown output: {}",
                        step.id,
                        ref_key
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Parse model type from string
pub fn parse_model_type(s: &str) -> Result<ModelType> {
    match s.to_lowercase().as_str() {
        "chat" => Ok(ModelType::Chat),
        "completion" => Ok(ModelType::Completion),
        "embedding" => Ok(ModelType::Embedding),
        "summarization" | "summary" => Ok(ModelType::Summarization),
        "classification" => Ok(ModelType::Classification),
        _ => Ok(ModelType::Custom(s.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_yaml_parsing() {
        let yaml = r#"
id: simple-chat
name: Simple Chat
description: A simple chat pipeline
steps:
  - id: generate
    name: Generate Response
    model: llama-2-7b-chat
    input:
      value: "Hello, world!"
    output: response
"#;

        let pipeline = PipelineDef::from_yaml_str(yaml).unwrap();
        assert_eq!(pipeline.id, "simple-chat");
        assert_eq!(pipeline.steps.len(), 1);
    }

    #[test]
    fn test_pipeline_json_parsing() {
        let json = r#"{
            "id": "simple-chat",
            "name": "Simple Chat",
            "description": "A simple chat pipeline",
            "steps": [
                {
                    "id": "generate",
                    "name": "Generate Response",
                    "model": "llama-2-7b-chat",
                    "input": {"value": "Hello, world!"},
                    "output": "response"
                }
            ]
        }"#;

        let pipeline = PipelineDef::from_json_str(json).unwrap();
        assert_eq!(pipeline.id, "simple-chat");
        assert_eq!(pipeline.steps.len(), 1);
    }

    #[test]
    fn test_multi_step_pipeline() {
        let yaml = r#"
id: summarization-pipeline
name: Summarization Pipeline
description: Multi-step summarization
steps:
  - id: extract
    name: Extract Key Points
    model_type: summarization
    input: "input"
    output: points
  - id: summarize
    name: Generate Summary
    model: llama-2-7b-chat
    input:
      template: "Key points: {{points}}"
    output: answer
"#;

        let pipeline = PipelineDef::from_yaml_str(yaml).unwrap();
        assert_eq!(pipeline.steps.len(), 2);
        assert_eq!(pipeline.steps[0].output, "points");
        assert_eq!(pipeline.steps[1].output, "answer");
    }

    #[test]
    fn test_validation_duplicate_ids() {
        let yaml = r#"
id: test
name: Test
description: Test
steps:
  - id: step1
    name: Step 1
    model: model1
    input: "input"
  - id: step1
    name: Step 2
    model: model2
    input: step1
"#;

        let pipeline = PipelineDef::from_yaml_str(yaml).unwrap();
        assert!(pipeline.validate().is_err());
    }

    #[test]
    fn test_model_type_parsing() {
        assert!(matches!(parse_model_type("chat").unwrap(), ModelType::Chat));
        assert!(matches!(
            parse_model_type("completion").unwrap(),
            ModelType::Completion
        ));
        assert!(matches!(
            parse_model_type("embedding").unwrap(),
            ModelType::Embedding
        ));
        assert!(matches!(
            parse_model_type("summarization").unwrap(),
            ModelType::Summarization
        ));
        assert!(matches!(
            parse_model_type("custom").unwrap(),
            ModelType::Custom(_)
        ));
    }
}
