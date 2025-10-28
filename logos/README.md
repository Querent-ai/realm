# Realm Logo Concepts

Holographic, retro-futuristic logo designs inspired by Jonny Quest aesthetics and the vision of inference orchestration.

## Vision

Realm orchestrates multiple isolated WASM sandboxes (realms) that share a single GPU through intelligent coordination. The logos capture:

- **Isolation**: Each realm is independent, sandboxed, secure
- **Orchestration**: Intelligent routing and coordination layer
- **Convergence**: All realms converge to share one powerful GPU
- **Holographic Tech**: Retro-futuristic wireframe aesthetic with scan lines
- **Quantum Harmony**: Multiple dimensions working in perfect sync

---

## Concept Designs

### 01 - Dimensional Portal
**File**: `concepts/01-dimensional-portal.svg`

**Vision**: A portal/gateway metaphor - 8 realm nodes arranged in a circle around a central GPU core. Energy streams flow from each realm into the shared resource.

**Elements**:
- 8 isolated realm nodes (R1-R8) in circular formation
- Central GPU core with power gradient
- Holographic rings showing dimensional layers
- Rotating scan beam
- Pulsing energy waves
- Dashed connection lines with animation

**Best for**: App icons, hero graphics, portal/gateway storytelling

---

### 02 - Orchestrator Nexus
**File**: `concepts/02-orchestrator-nexus.svg`

**Vision**: Top-down orchestration - a conductor node at the top distributes work to isolated hexagonal realms, which all converge to a powerful GPU triangle at the bottom.

**Elements**:
- Orchestrator conductor at top
- 5 hexagonal WASM sandbox nodes (isolation layer)
- GPU power triangle at bottom
- Mesh distribution network
- Flowing data particles
- Three-tier architecture visualization

**Best for**: Architecture diagrams, technical presentations, explaining the flow

---

### 03 - Quantum Weave
**File**: `concepts/03-quantum-weave.svg`

**Vision**: Quantum entanglement metaphor - 8 realm nodes orbit a central quantum core, connected by weave lines suggesting quantum entanglement and synchronized orchestration.

**Elements**:
- 8 realm nodes with orbital particles
- Central quantum core (GPU)
- Entanglement weave lines
- Expansion waves
- Particle animations
- Quantum grid background

**Best for**: Futuristic branding, sci-fi aesthetic, showing interconnection

---

## Design System

### Colors

**Primary Palette**:
- Cyan (`#00FFFF`) - Tech, orchestration, primary accent
- Deep Purple (`#6B46C1`, `#7C3AED`) - Realms, domains, authority
- Electric Blue (`#3B82F6`) - Technology, compute
- Cyber Gold (`#F59E0B`, `#FBBF24`) - GPU power, efficiency

**Usage**:
- Black background for maximum contrast
- Gradients for depth and holographic effect
- Glow filters for retro-futuristic aesthetic
- Animated scan lines and dashed lines

### Typography

- **Monospace fonts** for technical readout
- **UPPERCASE** for primary labels
- Small annotations for technical details

### Animation

All SVGs include:
- Pulsing nodes (varying opacity)
- Rotating scan beams
- Dashed line animations (marching ants)
- Expanding wave rings
- Particle motion paths

---

## Folder Structure

```
logos/
├── concepts/          # Main concept SVG designs
├── exports/           # Exported PNG/ICO files (various sizes)
├── assets/            # Supporting design assets
└── README.md          # This file
```

---

## Usage

### View in Browser
Open any `.svg` file directly in a web browser to see the animated version.

### Export to PNG
Use Inkscape, Adobe Illustrator, or online tools:
```bash
# Using Inkscape CLI
inkscape concepts/01-dimensional-portal.svg --export-png=exports/portal-1024.png --export-width=1024

# Multiple sizes
for size in 16 32 64 128 256 512 1024; do
  inkscape concepts/01-dimensional-portal.svg --export-png=exports/portal-${size}.png --export-width=${size}
done
```

### For Web
SVG files can be used directly:
```html
<img src="logos/concepts/01-dimensional-portal.svg" alt="Realm Logo" width="400">
```

### For Favicon
Export smallest sizes (16px, 32px, 48px) and convert to ICO:
```bash
convert exports/portal-16.png exports/portal-32.png exports/portal-48.png favicon.ico
```

---

## Next Steps

1. **Select Primary Logo**: Choose which concept best represents the vision
2. **Variations**: Create light/dark mode versions
3. **Simplified Icons**: Create simplified versions for small sizes
4. **Wordmark Integration**: Add "REALM" text lockup
5. **Export Assets**: Generate all required sizes and formats
6. **Brand Guidelines**: Document usage rules and specifications

---

## Design Philosophy

> "A logo should tell a story at a glance"

These designs capture the essence of Realm:
- **Not just branding** - they're visual explanations of the architecture
- **Holographic aesthetic** - retro-futuristic, technical without being sterile
- **Animated** - because orchestration is dynamic, not static
- **Dimensional** - showing layers: orchestration → isolation → convergence

Built with care by engineers who believe infrastructure should be beautiful. 🚀
