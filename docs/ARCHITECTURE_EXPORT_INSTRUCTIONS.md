# Architecture Diagram Export Instructions

## To Update the PNG Export

The `ARCHITECTURE.drawio` file has been created with a comprehensive architecture diagram. To export it to PNG:

### Option 1: Using draw.io Desktop App

1. Install draw.io from: <https://github.com/jgraph/drawio-desktop/releases>
2. Open `ARCHITECTURE.drawio`
3. File → Export As → PNG
4. Settings:
   - **Width:** 1920px (or higher for better quality)
   - **Transparent Background:** No
   - **Border Width:** 10px
   - **Scale:** 200% (for retina displays)
5. Save as `ARCHITECTURE.drawio.png`

### Option 2: Using draw.io Web

1. Go to: <https://app.diagrams.net/>
2. File → Open From → Device
3. Select `ARCHITECTURE.drawio`
4. File → Export as → PNG
5. Same settings as above
6. Save as `ARCHITECTURE.drawio.png`

### Option 3: Command Line (if you have drawio CLI)

```bash
# Install draw.io CLI
npm install -g @jgraph/drawio-cli

# Export to PNG
drawio -x -f png -o ARCHITECTURE.drawio.png -s 2 --width 1920 --border 10 ARCHITECTURE.drawio
```

## What's in the Diagram

The diagram shows:

### Main Layers

- **Tenant Layer** - 16 isolated WASM sandboxes (blue)
- **Host Runtime** - Shared infrastructure (orange)
  - Wasmtime (WASM host)
  - Memory64 Storage (purple)
  - Compute Backends (green)
  - Inference Flow (pink)

### Information Boxes

- Key Benefits
- Performance Metrics
- Economics
- Security Model
- Technology Stack
- Use Cases

### Visual Features

- Color-coded components for easy understanding
- Flow arrows showing data movement
- Size annotations (42KB WASM, 4.3GB model, etc.)
- 8-step inference flow
- 16 tenant isolation demonstration

## After Export

1. Verify the PNG looks good (text is readable, colors are correct)
2. Commit the updated PNG:

   ```bash
   git add ARCHITECTURE.drawio.png
   git commit -m "Update architecture diagram PNG export"
   ```

## Note

The current `ARCHITECTURE.drawio.png` may be from the old wasm-chord diagram. Please export a new one from the updated `ARCHITECTURE.drawio` file.
