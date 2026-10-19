PhotoCraft 0.10.0.3 (fork): prompted fills with the Infill brush and Generative Fill, running on your own computer.

## Downloads

- **Linux (Fedora and other distros):** `photocraft-0.10.0.3-linux-x86_64.AppImage`. Make it executable (`chmod +x`), then run it. It needs Fedora 40 or newer, or another distro with glibc 2.39 or newer.
- **Windows (x86_64):** `photocraft-0.10.0.3-windows-x86_64.zip`. Unzip it anywhere and run `photocraft.exe`, keeping the DLLs next to it. The build is unsigned, so Windows SmartScreen may ask you to confirm the first time.

## New

- **Infill brush** (in the Generative Brush box): paint over an area and, when you let go, a local Stable Diffusion model fills it from your prompt. The **Prompt** button in the options bar opens a menu with the prompt, negative prompt and seed; all three can be left empty. The same seed and prompt give the same result.
- **Edit › Generative Fill…**, also in the selection's right-click menu below Content-Aware Fill: fills the selection on a new layer named after the prompt.
  - The dialog has the prompt, negative prompt, seed and model.
  - **+** (or Enter in the prompt) makes a variation. It appears on the canvas as soon as it's ready; click another variation to switch to it.
  - **OK** keeps the variation shown, as one History step. **Cancel** takes it back.
- **Recommended models** for prompted fills, listed most accurate first: Realistic Vision 5.1, DreamShaper 8, AbsoluteReality, Stable Diffusion 2 and Stable Diffusion 1.5 inpainting. They are 2.1–2.6 GB each and use half-precision weights for the GPU. A full-precision Stable Diffusion 1.5 (4.3 GB) is included for computers without a usable GPU.
- **Reflection Removal models:** two dedicated reflection removers (DINOv3-S, MIT licence) are now recommended. Reflection Removal uses them before LaMa and MI-GAN.
- **Import › Choose Folder…** imports a Stable Diffusion ONNX pipeline folder. Hugging Face repositories laid out the same way import from a link.

## Changes from your 0.10.0.2 notes

- **Ctrl+click** a model's tick box to switch it on or off in every section it's listed in.
- The Generative Brush box shows the brushes as plain toolbar icons. Their names appear when you hover.

## Notes

- Prompted fills run Stable Diffusion 1.5 and 2 pipelines in ONNX format. Single-file checkpoints (.safetensors, most Civitai downloads) and Stable Diffusion XL can't run yet; the app says so if you pick one. The 0.10.0.2 recommendations that were checkpoints are replaced by the ONNX models above.
- A fill takes about 25 steps. On a GPU that's seconds; on a CPU it can take minutes. The progress window shows the time left.
