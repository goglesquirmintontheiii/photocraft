PhotoCraft 0.10.0.4 (fork): Generative Enhance and Sharpen, the Object Selection Brush, Stable Diffusion checkpoints and NPU support.

## Downloads

- **Linux (Fedora and other distros):** `photocraft-0.10.0.4-linux-x86_64.AppImage`. Make it executable (`chmod +x`), then run it. It needs Fedora 40 or newer, or another distro with glibc 2.39 or newer.
- **Windows (x86_64):** `photocraft-0.10.0.4-windows-x86_64.zip`. Unzip it anywhere and run `photocraft.exe`, keeping the DLLs next to it. The build is unsigned, so Windows SmartScreen may ask you to confirm the first time.

## New

- **Generative Enhance brush**, in the Generative Brush box: paint over a blurry or pixelated area to redraw it sharper.
  - The options bar has the model, **Detail** (Subtle, Medium or Strong) and the seed.
  - It uses a super-resolution model (Swin2SR, RealPLKSR or Real-ESRGAN), or a Stable Diffusion model run lightly.
- **Generative Sharpen…**, in the Edit menu and the selection's right-click menu below Generative Fill. It works like Generative Fill without a prompt: it sharpens the selection on a new layer.
  - **+** makes a variation. You choose the detail and the seed; seeds change the result with Stable Diffusion models.
- **Object Selection Brush**, in the Object Selection group (W): brush over an object and AI detection selects all of it. ⇧ adds to the selection, Alt subtracts.
- **Stable Diffusion checkpoints (.safetensors).** Most Civitai downloads now work, including Stable Diffusion 1.x, 2.x and **SDXL**.
  - A checkpoint is converted for PhotoCraft once, when it's imported or first used. The original file is deleted afterwards to save space.
  - The AI Models tab shows "Preparing it to run…" while that happens.
- **NPU support on Windows:** Edit › Preferences › AI Models › **Run models on** can be the graphics card, an NPU (AI accelerator, such as Intel AI Boost) or the processor. When a model can't run on your choice, it falls back to the next option.
- **More recommended models**, with the most accurate first in each section:
  - **Generative Enhance:** Swin2SR Real-World 4x, 4x Nomos2 RealPLKSR, RealPLKSR 4x and 2x, Swin2SR Classical 2x, Real-ESRGAN x4plus and Real-ESRGAN General v3.
  - **Object selection:** SAM 2.1 Large.
  - **Subject selection and matting:** BiRefNet HR Matting (2K), BiRefNet Matting, BiRefNet Dynamic and BiRefNet HRSOD.
  - **Prompted fills:** Realistic Vision 6.0 Inpainting (full and half size) and Stable Diffusion 2 Inpainting at full precision. These are checkpoints, prepared on first use.

## Also in this build

- The 38 latest upstream PhotoCraft changes, including Crop, Guides, layered TIFF and PSD export, and Actions fixes.
- **Sample All Layers** is now set separately for each tool, as in Photoshop, including the AI tools.

## Notes

- The checkpoint converter is checked against the reference Stable Diffusion code on small test models. Full-size checkpoints haven't been tried in this build environment, which can't download them.
- SDXL works at about 1024 px and needs much more memory than Stable Diffusion 1.5. A GPU with 8 GB or more is recommended.
- NPU support goes through DirectML, so it is Windows only. Some models may run faster on the GPU than on the NPU.
