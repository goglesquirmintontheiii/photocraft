PhotoCraft 0.10.0.2 (fork): the first AI tools, which run on your own computer.

## Downloads

- **Linux (Fedora and other distros):** `photocraft-0.10.0.2-linux-x86_64.AppImage`. Make it executable (`chmod +x`), then run it.
- **Windows (x86_64):** `photocraft-0.10.0.2-windows-x86_64.zip`. Unzip it anywhere and run `photocraft.exe`. Keep the DLLs next to it: they run the AI models on your graphics card. The build is unsigned, so Windows SmartScreen may ask you to confirm the first time.

## New tools

- **AI Object Selection** (in the Object Selection group, `W`): drag a box around something, or click it. A local AI model then selects it. Hold ⇧ to add to the selection, ⌥/Alt to subtract, or both to intersect. Choose the model in the options bar. It can use Segment Anything (SAM 2.1, MobileSAM) or a subject model such as BiRefNet.
- **Generative Brush** (the new button under the Brush Tool): click it to open a box of AI brushes beside the toolbar.
  - **Generative Removal:** paint over a person, object or blemish. When you let go, the model fills it in from what's around it. Works with LaMa or MI-GAN.
  - **Reflection Removal:** paint over a reflection to remove it. It uses a dedicated reflection model if you import one; otherwise it uses LaMa or MI-GAN.
- Both brushes can put the fill on a new layer and sample all layers. Every change is one step in History, and Undo takes it back.
- While a model runs, the area being filled stays tinted. The status bar and the progress window show progress, the time taken and the time left.
- Models run on the GPU where possible: DirectML on Windows, Vulkan (WebGPU) on Linux. If the GPU can't run a model, it runs on the CPU instead.

## Edit › Preferences › AI Models

- The tab is now called **AI Models**.
- **Multi-purpose models:** a model that does several jobs is listed in each of those sections. LaMa and MI-GAN appear under both Generative Removal and Reflection Removal; BiRefNet and RMBG under both Subject & Background and Object Selection.
  - Each section's tick box switches only that use on or off.
  - Rename (the pencil button), Delete and Download apply to the model everywhere.
- Disabled models are a little darker.
- **Import:** "Use as" accepts more than one use. It guesses from the link as you type, or from file names when you choose files. Click a use to change the guess.
- A recommended model you import from somewhere else, for example `Acly/MobileSAM` from Hugging Face, is recognised and shows as downloaded.
- **Accounts**, at the bottom of the tab, holds your Hugging Face and Civitai access tokens for models that need a sign-in. Tokens are stored on your computer only, readable only by you, and each is sent only to its own site.

## Notes

- Models run with ONNX Runtime 1.31. Stable Diffusion checkpoints (.safetensors) are for the Infill brush and Generative Fill, which come in the next release.
- No small, freely licensed reflection-removal model is available yet. Until one is, Reflection Removal fills the area like Generative Removal does.
