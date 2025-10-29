# Realm Logo - The Spiral 🌀

**Selected Logo**: The Spiral - Multiple paths converging to a single powerful core.

## Concept

The spiral represents:
- **Multiple entry points**: Each isolated workload/realm with its own unique path
- **Natural convergence**: Organic flow toward the center, like gravity or harmony
- **Single core**: The shared GPU at the center, powering everything
- **Dynamic motion**: Not static - shows active orchestration in progress

## Files

### Primary Versions

**spiral-primary.svg** (800x800)
- Full logo with wordmark
- Use for: Landing pages, hero sections, marketing materials
- Animated spirals with glow effects

**spiral-icon-only.svg** (400x400)
- Icon without text
- Use for: App icons, social media avatars, thumbnails
- Perfect for square spaces

**spiral-simple.svg** (400x400)
- Simplified version with fewer spirals
- Use for: Favicons, small UI elements, badges
- Optimized for small sizes (16px-64px)

**spiral-monochrome.svg** (400x400)
- White on black version
- Use for: Print materials, dark backgrounds, merchandise
- Works in single-color contexts

---

## Usage Guidelines

### ✅ DO:
- Use the simplified version for sizes below 64px
- Maintain the 6-spiral pattern in full versions
- Keep the glowing animated core
- Use on dark backgrounds for best effect
- Export to PNG for web use

### ❌ DON'T:
- Don't remove spirals or simplify the full version
- Don't change the gradient colors (cyan → purple → gold)
- Don't place on light backgrounds without adjustment
- Don't rotate or skew the logo
- Don't remove the glow effects

---

## Color Palette

**Spiral Colors** (in order from top):
1. Cyan: `#00FFFF` - Technology, orchestration
2. Blue: `#3B82F6` - Trust, compute
3. Purple: `#6B46C1` - Innovation, realms
4. Deep Purple: `#7C3AED` - Authority, power
5. Blue (repeat): `#3B82F6`
6. Cyan (repeat): `#00FFFF`

**Core Color**:
- Gold/Amber: `#F59E0B` - GPU power, convergence, value
- Light Gold: `#FBBF24` - Highlight ring

---

## Export Instructions

### For Web (PNG)
```bash
# High DPI versions
inkscape spiral-primary.svg --export-png=exports/png/spiral-primary-1024.png --export-width=1024
inkscape spiral-icon-only.svg --export-png=exports/png/spiral-icon-512.png --export-width=512
inkscape spiral-simple.svg --export-png=exports/png/spiral-simple-256.png --export-width=256

# Standard sizes
for size in 16 32 64 128 256; do
  inkscape spiral-simple.svg --export-png=exports/png/spiral-${size}.png --export-width=${size}
done
```

### For Favicon (ICO)
```bash
# Create multi-resolution favicon
convert exports/png/spiral-16.png exports/png/spiral-32.png exports/png/spiral-48.png exports/ico/favicon.ico
```

### For Social Media
```bash
# Twitter/X (400x400)
inkscape spiral-icon-only.svg --export-png=exports/png/spiral-twitter-400.png --export-width=400

# LinkedIn (300x300)
inkscape spiral-icon-only.svg --export-png=exports/png/spiral-linkedin-300.png --export-width=300

# GitHub Avatar (460x460)
inkscape spiral-icon-only.svg --export-png=exports/png/spiral-github-460.png --export-width=460

# Open Graph (1200x630)
# Use spiral-primary.svg with custom crop
```

---

## Technical Details

**Viewbox**: 400x400 (icon), 800x800 (primary)

**Animations**:
- Spiral opacity pulse: 2-2.4s cycles
- Core ring expansion: 2.5s cycle
- Core glow pulse: 2s cycle

**Filters**:
- Gaussian blur: 2.5-4px (size dependent)
- Used for holographic glow effect

**Gradients**:
- Spiral: Linear diagonal (cyan → purple → gold)
- Core: Radial (gold → purple → transparent)

---

## Metaphor & Meaning

The Spiral logo tells the Realm story:

> **"Many paths, naturally converging to one powerful center"**

- Each spiral = an isolated AI workload (realm)
- Different colors = different tenants/use cases
- The convergence = intelligent orchestration
- The glowing core = the shared GPU powering everything

It's not forced or artificial - it's natural, like water flowing downhill or planets orbiting a star. That's the beauty of Realm's architecture: it works WITH the nature of compute, not against it.

---

## Brand Voice Alignment

The spiral embodies Realm's brand:
- **Intelligent**: Natural, not forced
- **Efficient**: Direct paths, no waste
- **Powerful**: Strong central core
- **Dynamic**: Always in motion, always orchestrating
- **Beautiful**: Engineering that's aesthetically pleasing

---

Built with intention. Chosen with confidence. 🌀
