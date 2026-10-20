PhotoCraft 0.10.0.5 (fork): sharper Generative Removal and Enhance, a more precise Object Selection Brush, SDXL inpainting models, and a device choice for each model category.

## Downloads

- **Linux (Fedora and other distros):** `photocraft-0.10.0.5-linux-x86_64.AppImage`. Make it executable (`chmod +x`), then run it. It needs Fedora 40 or newer, or another distro with glibc 2.39 or newer.
- **Windows (x86_64):** `photocraft-0.10.0.5-windows-x86_64.zip`. Unzip it anywhere and run `photocraft.exe`, keeping the DLLs next to it. The build is unsigned, so Windows SmartScreen may ask you to confirm the first time.

## Improved

- **Generative Removal is crisper.** Where an object was removed, the fill now takes the fine texture and grain of its surroundings instead of leaving a soft patch. The model also sees the area at full resolution whenever the area is small enough, and loses less resolution when it's large.
- **Generative Enhance and Generative Sharpen keep detail.** They used to shrink the area to about a quarter of its size before redrawing it, which smoothed textures like brick walls more with each pass. The area now goes through the model at its own resolution, and the sharper result is brought back down to size. **Detail** changes how much of the result is used (Subtle, Medium), and Strong also redraws soft areas more.
- **Object Selection Brush:** it keeps only the objects the stroke touches. It used to sometimes add other objects elsewhere in the image, mostly on small images. The brushed area now shows in the accent colour, so it's easier to see.

## New

- **Run on, for each model category:** Edit › Preferences › AI Models. Open a category to choose where its models run: the graphics card, an NPU, the processor, or **As above** (the main **Run models on** choice).
- **More accurate Infill models**, at the top of Prompted fills:
  - **Fluently XL v3 Inpainting**
  - **Stable Diffusion XL Inpainting 0.1**

  Both are SDXL checkpoints (about 7 GB), prepared on first use. A GPU with 8 GB or more is recommended.
- More labels are translated: Subtle, Run on, As above, and the Object Selection Brush and Generative Enhance tool names.

## Also in this build

- The 76 latest upstream PhotoCraft changes, including Lens Blur depth from an alpha channel, Curves Ctrl/Cmd-click points, Save for Web keeping EXIF, print settings remembered, and faster PSD and PNG export.
