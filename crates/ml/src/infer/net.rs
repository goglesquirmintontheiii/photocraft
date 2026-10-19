//! A loaded ONNX network behind a narrow interface: named inputs in, named outputs out, plain
//! `Vec`s on both sides. The ONNX Runtime glue (through the `ort` crate) is compiled only with
//! the `onnx` (ONNX Runtime linked in, downloaded at build time) or `onnx-dynamic` (ONNX Runtime
//! loaded from a shared library at run time) feature; without either, loading a network says
//! the build has no AI runtime, and every tool reports that instead of crashing.

use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

/// An input's element type, as far as the models PhotoCraft runs need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F16,
    I64,
    I32,
    U8,
    Other,
}

/// A tensor's data. Inputs are converted to the type the network wants on the way in (`F32` data
/// for `F16` inputs or integer timesteps, `I64` data for `I32` token ids …); outputs of any
/// floating type come back as `F32`, of any integer type as `I64`.
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Data,
}

impl Tensor {
    pub fn f32(shape: &[i64], data: Vec<f32>) -> Tensor {
        Tensor { shape: shape.to_vec(), data: Data::F32(data) }
    }
    pub fn i64(shape: &[i64], data: Vec<i64>) -> Tensor {
        Tensor { shape: shape.to_vec(), data: Data::I64(data) }
    }
    pub fn u8(shape: &[i64], data: Vec<u8>) -> Tensor {
        Tensor { shape: shape.to_vec(), data: Data::U8(data) }
    }

    /// The values as `f32`, whatever they were.
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            Data::F32(v) => v.clone(),
            Data::I64(v) => v.iter().map(|x| *x as f32).collect(),
            Data::U8(v) => v.iter().map(|x| f32::from(*x)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named input or output: element type and shape (-1 for a dimension decided at run time).
#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    pub name: String,
    pub elem: Elem,
    pub shape: Vec<i64>,
}

/// What a network needs and gives.
pub trait Network: Send + Sync {
    fn inputs(&self) -> Vec<Port>;
    fn outputs(&self) -> Vec<Port>;
    /// Runs the network. Inputs are matched by name; every output comes back.
    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>>;
    /// Where it runs: "DirectML", "WebGPU", "CPU" …
    fn backend(&self) -> String;
}

/// Loads networks (the real runtime, or a test double).
pub trait Loader: Send + Sync {
    fn load(&self, path: &Path) -> Result<Arc<dyn Network>>;
}

/// Loaded networks are kept: loading a 300 MB encoder takes seconds, and the tools run the same
/// model again and again. A few at a time (a diffusion pipeline is four), the oldest dropped first.
const KEEP: usize = 6;

struct Cache {
    nets: Vec<(PathBuf, Arc<dyn Network>)>,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { nets: Vec::new() });

/// The network in `path`, loaded once with `loader`.
pub fn cached(loader: &dyn Loader, path: &Path) -> Result<Arc<dyn Network>> {
    {
        let mut c = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = c.nets.iter().position(|(p, _)| p == path) {
            let hit = c.nets.remove(i);
            let net = hit.1.clone();
            c.nets.push(hit);
            return Ok(net);
        }
    }
    let net = loader.load(path)?;
    let mut c = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    c.nets.retain(|(p, _)| p != path);
    c.nets.push((path.to_path_buf(), net.clone()));
    while c.nets.len() > KEEP {
        c.nets.remove(0);
    }
    Ok(net)
}

/// Forgets loaded networks (after a model is deleted, or to free memory).
pub fn clear_cache() {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner).nets.clear();
}

/// Forgets the networks loaded from files under `dir` (a model's folder).
pub fn forget(dir: &Path) {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner).nets.retain(|(p, _)| !p.starts_with(dir));
}

/// Forgets the loaded networks whose file `drop` picks (another diffusion pipeline's parts).
pub fn forget_where(drop: impl Fn(&Path) -> bool) {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner).nets.retain(|(p, _)| !drop(p));
}

/// The ONNX Runtime loader for this build.
pub fn runtime() -> &'static dyn Loader {
    &ort_impl::RUNTIME
}

/// Whether this build can run models at all.
pub fn available() -> bool {
    ort_impl::AVAILABLE
}

#[cfg(all(not(target_arch = "wasm32"), any(feature = "onnx", feature = "onnx-dynamic")))]
mod ort_impl {
    use super::*;
    use ort::ep::{self, ExecutionProviderDispatch};
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::value::{DynValue, TensorElementType, ValueType};
    use std::sync::OnceLock;

    pub const AVAILABLE: bool = true;
    pub static RUNTIME: Ort = Ort;

    pub struct Ort;

    fn err(context: &str, e: impl std::fmt::Display) -> Error {
        Error::Unsupported(format!("{context}: {e}"))
    }

    /// The ONNX Runtime environment, made once per process.
    fn environment() -> Result<&'static ort::environment::Environment> {
        static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
        ENV.get_or_init(|| {
            #[cfg(feature = "onnx-dynamic")]
            let builder = ort::init_from(library_path()).map_err(|e| format!("Couldn't load ONNX Runtime ({}): {e}", library_path().display()))?;
            #[cfg(not(feature = "onnx-dynamic"))]
            let builder = ort::init();
            builder.with_name("photocraft").build().map_err(|e| format!("Couldn't start ONNX Runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| Error::Unsupported(e.clone()))
    }

    /// Where `onnx-dynamic` builds find ONNX Runtime: `PHOTOCRAFT_ONNXRUNTIME`, else beside the
    /// executable (or in `../lib`, as in the AppImage).
    #[cfg(feature = "onnx-dynamic")]
    fn library_path() -> PathBuf {
        if let Some(p) = std::env::var_os("PHOTOCRAFT_ONNXRUNTIME") {
            return PathBuf::from(p);
        }
        let name = if cfg!(windows) {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)).unwrap_or_default();
        [exe_dir.join(name), exe_dir.join("../lib").join(name)].into_iter().find(|p| p.exists()).unwrap_or_else(|| exe_dir.join(name))
    }

    /// GPU execution providers this build has, best first.
    fn gpu_providers() -> Vec<(&'static str, ExecutionProviderDispatch)> {
        #[allow(unused_mut)]
        let mut v: Vec<(&'static str, ExecutionProviderDispatch)> = Vec::new();
        #[cfg(feature = "directml")]
        v.push(("DirectML", ep::DirectML::default().build().error_on_failure()));
        #[cfg(feature = "cuda")]
        v.push(("CUDA", ep::CUDA::default().build().error_on_failure()));
        #[cfg(feature = "webgpu")]
        v.push(("WebGPU", ep::WebGPU::default().build().error_on_failure()));
        v
    }

    fn elem(t: TensorElementType) -> Elem {
        match t {
            TensorElementType::Float32 => Elem::F32,
            TensorElementType::Float16 => Elem::F16,
            TensorElementType::Int64 => Elem::I64,
            TensorElementType::Int32 => Elem::I32,
            TensorElementType::Uint8 => Elem::U8,
            _ => Elem::Other,
        }
    }

    fn port(name: &str, ty: &ValueType) -> Port {
        match ty {
            ValueType::Tensor { ty, shape, .. } => Port { name: name.to_string(), elem: elem(*ty), shape: shape.to_vec() },
            _ => Port { name: name.to_string(), elem: Elem::Other, shape: Vec::new() },
        }
    }

    pub struct OrtNet {
        path: PathBuf,
        /// The session and where it runs; a GPU session that fails to run is swapped for a CPU one.
        session: Mutex<(Session, &'static str)>,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    }

    fn cpu_session(env: &ort::environment::Environment, path: &Path) -> ort::Result<Session> {
        Session::builder(env)
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort::Error::from))
            .and_then(|b| b.with_execution_providers([ep::CPU::default().build()]).map_err(ort::Error::from))
            .and_then(|mut b| b.commit_from_file(path))
    }

    impl Loader for Ort {
        fn load(&self, path: &Path) -> Result<Arc<dyn Network>> {
            let env = environment()?;
            let mut last = String::new();
            let mut found: Option<(Session, &'static str)> = None;
            if crate::infer::gpu_allowed() {
                for (name, provider) in gpu_providers() {
                    let attempt = Session::builder(env)
                        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort::Error::from))
                        .and_then(|b| b.with_memory_pattern(false).map_err(ort::Error::from))
                        .and_then(|b| b.with_parallel_execution(false).map_err(ort::Error::from))
                        .and_then(|b| b.with_execution_providers([provider]).map_err(ort::Error::from))
                        .and_then(|mut b| b.commit_from_file(path));
                    match attempt {
                        Ok(s) => {
                            found = Some((s, name));
                            break;
                        }
                        Err(e) => {
                            log::info!("{name} can't run {}: {e}; trying the next option", path.display());
                            last = e.to_string();
                        }
                    }
                }
            }
            let (session, backend) = match found {
                Some(f) => f,
                None => {
                    let s = cpu_session(env, path).map_err(|e| {
                        let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        Error::Invalid(format!(
                            "{file} couldn't be loaded as an ONNX model: {e}{}",
                            if last.is_empty() { String::new() } else { format!(" ({last})") }
                        ))
                    })?;
                    (s, "CPU")
                }
            };
            log::info!("loaded {} on {backend}", path.display());
            let inputs = session.inputs().iter().map(|o| port(o.name(), o.dtype())).collect();
            let outputs = session.outputs().iter().map(|o| port(o.name(), o.dtype())).collect();
            Ok(Arc::new(OrtNet { path: path.to_path_buf(), session: Mutex::new((session, backend)), inputs, outputs }))
        }
    }

    impl OrtNet {
        fn collect(&self, out: &ort::session::SessionOutputs<'_>) -> Vec<(String, Tensor)> {
            self.outputs.iter().filter_map(|p| out.get(p.name.as_str()).and_then(from_value).map(|t| (p.name.clone(), t))).collect()
        }
    }

    fn to_value(t: Tensor, want: Elem) -> ort::Result<DynValue> {
        let shape: Vec<usize> = t.shape.iter().map(|d| (*d).max(0) as usize).collect();
        let f16 = |v: Vec<f32>| v.into_iter().map(half::f16::from_f32).collect::<Vec<_>>();
        Ok(match (t.data, want) {
            (Data::F32(v), Elem::F16) => ort::value::Tensor::<half::f16>::from_array((shape, f16(v)))?.into_dyn(),
            (Data::F32(v), Elem::I64) => ort::value::Tensor::<i64>::from_array((shape, v.into_iter().map(|x| x.round() as i64).collect::<Vec<_>>()))?.into_dyn(),
            (Data::F32(v), Elem::I32) => ort::value::Tensor::<i32>::from_array((shape, v.into_iter().map(|x| x.round() as i32).collect::<Vec<_>>()))?.into_dyn(),
            (Data::F32(v), _) => ort::value::Tensor::<f32>::from_array((shape, v))?.into_dyn(),
            (Data::I64(v), Elem::I32) => {
                ort::value::Tensor::<i32>::from_array((shape, v.into_iter().map(|x| x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32).collect::<Vec<_>>()))?.into_dyn()
            }
            (Data::I64(v), Elem::F32) => ort::value::Tensor::<f32>::from_array((shape, v.into_iter().map(|x| x as f32).collect::<Vec<_>>()))?.into_dyn(),
            (Data::I64(v), Elem::F16) => ort::value::Tensor::<half::f16>::from_array((shape, f16(v.into_iter().map(|x| x as f32).collect())))?.into_dyn(),
            (Data::I64(v), _) => ort::value::Tensor::<i64>::from_array((shape, v))?.into_dyn(),
            (Data::U8(v), _) => ort::value::Tensor::<u8>::from_array((shape, v))?.into_dyn(),
        })
    }

    fn from_value(v: &DynValue) -> Option<Tensor> {
        if let Ok((s, d)) = v.try_extract_tensor::<f32>() {
            return Some(Tensor::f32(s, d.to_vec()));
        }
        if let Ok((s, d)) = v.try_extract_tensor::<half::f16>() {
            return Some(Tensor::f32(s, d.iter().map(|x| x.to_f32()).collect()));
        }
        if let Ok((s, d)) = v.try_extract_tensor::<u8>() {
            return Some(Tensor::u8(s, d.to_vec()));
        }
        if let Ok((s, d)) = v.try_extract_tensor::<i64>() {
            return Some(Tensor::i64(s, d.to_vec()));
        }
        if let Ok((s, d)) = v.try_extract_tensor::<i32>() {
            return Some(Tensor::i64(s, d.iter().map(|x| i64::from(*x)).collect()));
        }
        None
    }

    impl Network for OrtNet {
        fn inputs(&self) -> Vec<Port> {
            self.inputs.clone()
        }
        fn outputs(&self) -> Vec<Port> {
            self.outputs.clone()
        }
        fn backend(&self) -> String {
            self.session.lock().unwrap_or_else(PoisonError::into_inner).1.to_string()
        }
        fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>> {
            let make = |inputs: &[(String, Tensor)]| -> Result<Vec<(String, DynValue)>> {
                let mut values = Vec::with_capacity(inputs.len());
                for (name, t) in inputs {
                    let want = self.inputs.iter().find(|p| &p.name == name).map_or(Elem::F32, |p| p.elem);
                    values.push((name.clone(), to_value(t.clone(), want).map_err(|e| err(&format!("The model's input {name} couldn't be made"), e))?));
                }
                Ok(values)
            };
            let mut guard = self.session.lock().unwrap_or_else(PoisonError::into_inner);
            let first = {
                let (session, _) = &mut *guard;
                session.run(make(&inputs)?).map(|out| self.collect(&out))
            };
            match first {
                Ok(r) => Ok(r),
                Err(e) if guard.1 != "CPU" => {
                    // A GPU that loaded the model but can't run it: run it on the CPU from now on.
                    log::warn!("{} failed on {} ({e}); running it on the CPU instead", self.path.display(), guard.1);
                    let cpu = cpu_session(environment()?, &self.path).map_err(|e| Error::Invalid(format!("The model couldn't run: {e}")))?;
                    *guard = (cpu, "CPU");
                    let (session, _) = &mut *guard;
                    let out = session.run(make(&inputs)?).map_err(|e| Error::Invalid(format!("The model couldn't run: {e}")))?;
                    Ok(self.collect(&out))
                }
                Err(e) => Err(Error::Invalid(format!("The model couldn't run: {e}"))),
            }
        }
    }
}

#[cfg(not(all(not(target_arch = "wasm32"), any(feature = "onnx", feature = "onnx-dynamic"))))]
mod ort_impl {
    use super::*;

    pub const AVAILABLE: bool = false;
    pub static RUNTIME: NoRuntime = NoRuntime;

    pub struct NoRuntime;

    impl Loader for NoRuntime {
        fn load(&self, _: &Path) -> Result<Arc<dyn Network>> {
            Err(Error::Unsupported(if cfg!(target_arch = "wasm32") {
                "AI models run in the desktop app.".into()
            } else {
                "This build of PhotoCraft has no AI model runtime. Download a release build to use AI tools.".into()
            }))
        }
    }
}
