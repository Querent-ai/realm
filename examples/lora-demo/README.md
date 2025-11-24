# LoRA Adapter Demo

This example demonstrates how to use LoRA (Low-Rank Adaptation) adapters with Realm:

1. **Create a LoRA adapter** with custom weights
2. **Load it into RuntimeManager**
3. **Set it for a tenant**
4. **Generate text with LoRA applied**

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to:
- Fine-tune models for specific tasks without storing full model copies
- Apply different adapters per tenant for multi-tenant scenarios
- Reduce memory usage (adapters are typically <1% of model size)

## Usage

```bash
cargo run --release --bin lora-demo -- <model_path>
```

### Example

```bash
cargo run --release --bin lora-demo -- ~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
```

## What This Demo Shows

### Step 1: Create LoRA Adapter
- Creates a LoRA adapter with rank=8, alpha=16.0
- Adds dummy weights for layer 0 attention (query, key, value, output)
- In production, you would load weights from a trained LoRA file

### Step 2: Load Adapter
- Registers the adapter in RuntimeManager
- Makes it available for use during inference

### Step 3: Set Default Model
- Configures the base model for inference
- Model will be loaded when first used

### Step 4: Set Adapter for Tenant
- Associates the LoRA adapter with a specific tenant
- Each tenant can have their own adapter

### Step 5: Generate with LoRA
- Generates text using the base model + LoRA adapter
- LoRA weights are automatically applied during forward pass

## Expected Output

```
🎯 LoRA Adapter Demo

📦 Model: /path/to/model.gguf

🔧 Creating RuntimeManager...
✅ RuntimeManager created

📝 Step 1: Creating LoRA adapter...
✅ LoRA adapter created:
   - ID: demo_adapter
   - Rank: 8
   - Alpha: 16.0
   - Scale: 2.0
   - Layers: layer.0 (attention weights)

📥 Step 2: Loading LoRA adapter into RuntimeManager...
✅ LoRA adapter loaded

🤖 Step 3: Setting default model...
✅ Default model set

👤 Step 4: Setting LoRA adapter for tenant...
✅ LoRA adapter set for tenant: demo_tenant

🚀 Step 5: Generating text with LoRA applied...
   Prompt: What is the capital of France?

✅ Generation successful!
   Response: Paris

🎉 LoRA adapter is being applied during inference!
```

## Production Usage

In production, you would:

1. **Train a LoRA adapter** on your specific task/dataset
2. **Save it** in a standard format (e.g., safetensors)
3. **Load it** from file instead of creating dummy weights
4. **Apply it** to fine-tune the model for your use case

## Notes

- This demo uses dummy weights for demonstration
- Real LoRA adapters are trained on specific tasks
- LoRA supports all quantization formats (automatically dequantized)
- Each tenant can have their own adapter for multi-tenant scenarios

