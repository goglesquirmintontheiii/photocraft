PhotoCraft 0.10.0 plus the first batch of local AI work (phase 1 of 3): the model manager.

**New: Edit › Preferences › AI** (below Integrations; also Edit › Preferences › AI Models…)

- One collapsible section per kind of model (collapsed until opened or matched by a search): Object Selection, Subject & Background, Generative Removal, Reflection Removal, Generative Fill & Infill, and Other Models.
- Tick a model to enable or disable it, see its size on disk, delete it with the trash button (asks first).
- 12 recommended models with one-click Download: SAM 2.1 Base+/Small/Tiny, MobileSAM, BiRefNet, BiRefNet Lite, BRIA RMBG 1.4, LaMa, MI-GAN, SDXL Inpainting 0.1, Stable Diffusion 2 Inpainting and Stable Diffusion 1.5 Inpainting. Recommended rows are darker and striped; once downloaded they turn the normal colour and keep their stripes.
- Import from a file, or from a link: a Hugging Face file or repository (`org/name` works too), a Civitai model page, or any direct download link.
- Downloads and imports run in the background with a progress bar, size done, time taken and an estimate of the time left. They resume after an interruption and are checked against the checksum Hugging Face and Civitai publish.
- Models are stored in `~/.config/photocraft/Models` with your other PhotoCraft files.

The tools that use these models (AI Object Selection, the Generative brushes and Generative Fill) arrive in the next two batches. Nothing in this batch runs a model yet.

**Install on Fedora:** download the `.AppImage`, then `chmod +x photocraft-*.AppImage` and run it. If it says FUSE is missing: `sudo dnf install fuse fuse-libs`.
