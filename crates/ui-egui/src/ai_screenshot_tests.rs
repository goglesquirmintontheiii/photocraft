//! Pictures of the AI tools for review (ignored; run with PHOTOCRAFT_AI_SHOTS=<folder>, and for
//! real variations a model runtime: `--features photocraft-ml/onnx-dynamic` and
//! PHOTOCRAFT_ONNXRUNTIME): the Generative Brush box, Infill's options bar and prompt menu, and
//! the Generative Fill dialog with its variations.

use egui::{Event, Modifiers, PointerButton, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

fn models() -> photocraft_ml::Manager {
    let dir = std::env::temp_dir().join(format!("photocraft-ai-shots-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let m = photocraft_ml::Manager::open(&dir).unwrap();
    let pipeline = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ml/tests/fixtures/tiny_sd");
    m.import_folder(&pipeline, &[photocraft_ml::ModelKind::Inpainting], Some("Realistic Vision 5.1 Inpainting".into())).unwrap();
    while m.any_active() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    m
}

fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
    let mut ready = false;
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).wgpu().build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::Studio);
                ready = true;
                return;
            }
            let ctx = ui.ctx().clone();
            crate::panels::title_bar(app, ui);
            crate::panels::options_bar(app, ui);
            crate::panels::toolbar(app, ui);
            crate::panels::right_dock(app, ui);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            crate::dialogs::show(app, &ctx);
            app.session.poll_jobs();
        },
        app,
    );
    h.run_steps(6);
    h
}

fn click(h: &mut Harness<'static, PhotocraftApp>, at: egui::Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(2);
}

#[test]
#[ignore = "writes pictures for review: PHOTOCRAFT_AI_SHOTS=<folder>"]
fn ai_tools_pictures() {
    let Some(dir) = std::env::var_os("PHOTOCRAFT_AI_SHOTS").map(std::path::PathBuf::from) else { return };
    std::fs::create_dir_all(&dir).unwrap();
    let _gpu = photocraft_testkit::gpu_lock();
    let m = models();
    let services = crate::Services { models: Some(m), ..Default::default() };
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    app.run("file.new", json!({"width": 640, "height": 420})).unwrap();
    app.run("layer.new.layer", json!({"name": "Photo"})).unwrap();
    app.run("edit.fill", json!({"color": "#7fa7c9"})).unwrap();
    app.run("select.rect", json!({"x": 0, "y": 260, "width": 640, "height": 160})).unwrap();
    app.run("edit.fill", json!({"color": "#6c8f4e"})).unwrap();
    app.run("select.rect", json!({"x": 220, "y": 150, "width": 200, "height": 180})).unwrap();
    app.ui.tool = Tool::GenerativeInfill;
    app.ui.tool_options.infill_prompt = "a red bicycle leaning on a fence".into();
    let mut h = harness(app);

    // The Generative Brush box, opened from the toolbar.
    let button = h.get_by_label("Generative Brush").rect().center();
    click(&mut h, button);
    h.render().unwrap().save(dir.join("generative-brush-box.png")).unwrap();
    h.hover_at(egui::pos2(700.0, 400.0));
    h.run_steps(2);
    h.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);

    // Infill's options bar and its prompt menu.
    let prompt = h.get_by_label_contains("a red bicycle").rect().center();
    click(&mut h, prompt);
    h.render().unwrap().save(dir.join("infill-prompt-menu.png")).unwrap();
    h.event(Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);

    // Generative Fill: the dialog, then with variations when a runtime can make them.
    crate::generative_fill_ui::open(h.state_mut()).unwrap();
    h.run_steps(4);
    h.render().unwrap().save(dir.join("generative-fill-dialog.png")).unwrap();
    if photocraft_ml::infer::runtime_available() && std::env::var_os("PHOTOCRAFT_ONNXRUNTIME").is_some() {
        photocraft_ml::infer::set_gpu_allowed(false);
        for _ in 0..3 {
            h.state_mut().run("edit.generativeFill", json!({"prompt": "a red bicycle", "steps": 2, "preview": true})).unwrap();
            for _ in 0..400 {
                h.run_steps(1);
                if h.state().session.active_job().is_none() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        h.run_steps(4);
        assert_eq!(h.state().session.generative_fill_previews().len(), 3);
        h.render().unwrap().save(dir.join("generative-fill-variations.png")).unwrap();
    }
}
