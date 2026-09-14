use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use ort::{
    ep::{CPU, CUDA},
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::sync::Mutex;

const INPUT_NAME: &str = "input";
const DESCRIPTOR_OUTPUT: &str = "output_feats";
const KEYPOINT_OUTPUT: &str = "output_keypoints";
const RELIABILITY_OUTPUT: &str = "output_heatmap";
const DESCRIPTOR_DIM: usize = 64;
const KEYPOINT_CHANNELS: usize = 65;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseFeature {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub descriptor: Vec<f32>,
}

pub struct XfeatExtractor {
    inference_session: Mutex<Session>,
    model_width: i32,
    model_height: i32,
}

#[derive(Debug, Clone, Copy)]
struct TensorLayout {
    height: usize,
    width: usize,
    channels: usize,
    channels_first: bool,
}

impl TensorLayout {
    fn parse(shape: &[i64], channels: usize, name: &str) -> Result<Self, String> {
        if shape.len() != 4 || shape[0] != 1 {
            return Err(format!("{name} must be a batch=1 4D tensor, got {shape:?}"));
        }
        let channels = channels as i64;
        let (height, width, channels_first) = if shape[1] == channels {
            (shape[2], shape[3], true)
        } else if shape[3] == channels {
            (shape[1], shape[2], false)
        } else {
            return Err(format!("{name} missing {channels} channels, got {shape:?}"));
        };
        if height <= 0 || width <= 0 {
            return Err(format!("{name} has invalid spatial dimensions: {shape:?}"));
        }
        Ok(Self {
            height: height as usize,
            width: width as usize,
            channels: channels as usize,
            channels_first,
        })
    }

    fn expected_len(self) -> usize {
        self.height * self.width * self.channels
    }

    fn index(self, y: usize, x: usize, channel: usize) -> usize {
        if self.channels_first {
            (channel * self.height + y) * self.width + x
        } else {
            (y * self.width + x) * self.channels + channel
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Letterbox {
    scale: f32,
    offset_x: i32,
    offset_y: i32,
    source_width: i32,
    source_height: i32,
}

impl XfeatExtractor {
    pub fn new<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self, String> {
        crate::self_heal_load_onnx_dylib();
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_execution_providers([CUDA::default().build(), CPU::default().build()])
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(1)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;
        validate_model_names(&session)?;
        Ok(Self {
            inference_session: Mutex::new(session),
            model_width: 640,
            model_height: 640,
        })
    }

    pub fn extract_features(
        &self,
        input_image: &Mat,
        max_keypoints: usize,
    ) -> Result<Vec<SparseFeature>, String> {
        if max_keypoints == 0 {
            return Ok(Vec::new());
        }
        validate_image(input_image)?;
        let (adapted, letterbox) =
            letterbox_gray(input_image, self.model_width, self.model_height)?;
        let tensor = normalize_gray(&adapted)?;
        let input = Value::from_array((
            [
                1usize,
                1,
                self.model_height as usize,
                self.model_width as usize,
            ],
            tensor,
        ))
        .map_err(|e| e.to_string())?;

        let mut session = self.inference_session.lock().map_err(|e| e.to_string())?;
        let outputs = session
            .run(ort::inputs![INPUT_NAME => input])
            .map_err(|e| e.to_string())?;
        let (descriptor_shape, descriptor_data) = named_tensor(&outputs, DESCRIPTOR_OUTPUT)?;
        let (keypoint_shape, keypoint_data) = named_tensor(&outputs, KEYPOINT_OUTPUT)?;
        let (reliability_shape, reliability_data) = named_tensor(&outputs, RELIABILITY_OUTPUT)?;

        extract_sparse_features(
            descriptor_shape,
            descriptor_data,
            keypoint_shape,
            keypoint_data,
            reliability_shape,
            reliability_data,
            self.model_width as usize,
            self.model_height as usize,
            letterbox,
            max_keypoints,
        )
    }
}

fn validate_model_names(session: &Session) -> Result<(), String> {
    for input in [INPUT_NAME] {
        if !session.inputs().iter().any(|outlet| outlet.name() == input) {
            return Err(format!("XFeat model missing input {input}"));
        }
    }
    for output in [DESCRIPTOR_OUTPUT, KEYPOINT_OUTPUT, RELIABILITY_OUTPUT] {
        if !session
            .outputs()
            .iter()
            .any(|outlet| outlet.name() == output)
        {
            return Err(format!("XFeat model missing output {output}"));
        }
    }
    Ok(())
}

fn named_tensor<'output>(
    outputs: &'output ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(&'output [i64], &'output [f32]), String> {
    let output = outputs
        .get(name)
        .ok_or_else(|| format!("XFeat inference missing named output {name}"))?;
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    Ok((shape, data))
}

fn validate_image(image: &Mat) -> Result<(), String> {
    if image.empty() || image.rows() <= 0 || image.cols() <= 0 {
        return Err("XFeat input image is empty".to_string());
    }
    if image.depth() != core::CV_8U {
        return Err(format!(
            "XFeat only accepts CV_8U images, got depth={}",
            image.depth()
        ));
    }
    if !matches!(image.channels(), 1 | 3 | 4) {
        return Err(format!(
            "XFeat only accepts gray/BGR/BGRA images, got channels={}",
            image.channels()
        ));
    }
    Ok(())
}

fn letterbox_gray(image: &Mat, target_w: i32, target_h: i32) -> Result<(Mat, Letterbox), String> {
    let mut gray = Mat::default();
    match image.channels() {
        1 => gray = image.clone(),
        3 => imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)
            .map_err(|e| e.to_string())?,
        4 => imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGRA2GRAY, 0)
            .map_err(|e| e.to_string())?,
        _ => unreachable!(),
    }

    let scale = (target_w as f32 / image.cols() as f32).min(target_h as f32 / image.rows() as f32);
    let resized_w = ((image.cols() as f32 * scale).round() as i32).clamp(1, target_w);
    let resized_h = ((image.rows() as f32 * scale).round() as i32).clamp(1, target_h);
    let mut resized = Mat::default();
    imgproc::resize(
        &gray,
        &mut resized,
        Size::new(resized_w, resized_h),
        0.0,
        0.0,
        if scale < 1.0 {
            imgproc::INTER_AREA
        } else {
            imgproc::INTER_LINEAR
        },
    )
    .map_err(|e| e.to_string())?;

    let offset_x = (target_w - resized_w) / 2;
    let offset_y = (target_h - resized_h) / 2;
    let mut padded = Mat::default();
    core::copy_make_border(
        &resized,
        &mut padded,
        offset_y,
        target_h - resized_h - offset_y,
        offset_x,
        target_w - resized_w - offset_x,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )
    .map_err(|e| e.to_string())?;
    Ok((
        padded,
        Letterbox {
            scale,
            offset_x,
            offset_y,
            source_width: image.cols(),
            source_height: image.rows(),
        },
    ))
}

fn normalize_gray(gray: &Mat) -> Result<Vec<f32>, String> {
    let count = (gray.rows() * gray.cols()) as usize;
    let mut values = Vec::with_capacity(count);
    let mut sum = 0.0f64;
    for y in 0..gray.rows() {
        let row = gray.ptr(y).map_err(|e| e.to_string())?;
        let row = unsafe { std::slice::from_raw_parts(row, gray.cols() as usize) };
        for &value in row {
            let value = value as f32;
            values.push(value);
            sum += value as f64;
        }
    }
    let mean = (sum / count as f64) as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / count as f32;
    let denominator = variance.sqrt().max(1e-6);
    for value in &mut values {
        *value = (*value - mean) / denominator;
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn extract_sparse_features(
    descriptor_shape: &[i64],
    descriptors: &[f32],
    keypoint_shape: &[i64],
    keypoints: &[f32],
    reliability_shape: &[i64],
    reliability: &[f32],
    model_width: usize,
    model_height: usize,
    letterbox: Letterbox,
    limit: usize,
) -> Result<Vec<SparseFeature>, String> {
    let descriptor_layout =
        TensorLayout::parse(descriptor_shape, DESCRIPTOR_DIM, DESCRIPTOR_OUTPUT)?;
    let keypoint_layout = TensorLayout::parse(keypoint_shape, KEYPOINT_CHANNELS, KEYPOINT_OUTPUT)?;
    let reliability_layout = TensorLayout::parse(reliability_shape, 1, RELIABILITY_OUTPUT)?;
    if descriptor_layout.height != keypoint_layout.height
        || descriptor_layout.width != keypoint_layout.width
        || reliability_layout.height != keypoint_layout.height
        || reliability_layout.width != keypoint_layout.width
        || keypoint_layout.height * 8 != model_height
        || keypoint_layout.width * 8 != model_width
    {
        return Err(format!(
            "XFeat output spatial dimensions mismatch: descriptor={descriptor_shape:?}, keypoint={keypoint_shape:?}, reliability={reliability_shape:?}"
        ));
    }
    for (name, actual, expected) in [
        (
            DESCRIPTOR_OUTPUT,
            descriptors.len(),
            descriptor_layout.expected_len(),
        ),
        (
            KEYPOINT_OUTPUT,
            keypoints.len(),
            keypoint_layout.expected_len(),
        ),
        (
            RELIABILITY_OUTPUT,
            reliability.len(),
            reliability_layout.expected_len(),
        ),
    ] {
        if actual != expected {
            return Err(format!(
                "{name} element count mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    if descriptors
        .iter()
        .chain(keypoints)
        .chain(reliability)
        .any(|v| !v.is_finite())
    {
        return Err("XFeat output contains NaN or infinity".to_string());
    }

    let mut keypoint_score_map = vec![0.0f32; model_width * model_height];
    let mut reliability_score_map = vec![0.0f32; model_width * model_height];
    for cell_y in 0..keypoint_layout.height {
        for cell_x in 0..keypoint_layout.width {
            let mut max_logit = f32::NEG_INFINITY;
            for channel in 0..KEYPOINT_CHANNELS {
                max_logit =
                    max_logit.max(keypoints[keypoint_layout.index(cell_y, cell_x, channel)]);
            }
            let mut exponentials = [0.0f32; KEYPOINT_CHANNELS];
            let mut sum = 0.0f32;
            for channel in 0..KEYPOINT_CHANNELS {
                let value =
                    (keypoints[keypoint_layout.index(cell_y, cell_x, channel)] - max_logit).exp();
                exponentials[channel] = value;
                sum += value;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err("XFeat keypoint softmax normalization failed".to_string());
            }
            for local_y in 0..8 {
                for local_x in 0..8 {
                    let x = cell_x * 8 + local_x;
                    let y = cell_y * 8 + local_y;
                    let reliability_score = bilinear_channel(
                        reliability,
                        reliability_layout,
                        x as f32 * reliability_layout.width as f32 / model_width as f32 - 0.5,
                        y as f32 * reliability_layout.height as f32 / model_height as f32 - 0.5,
                        0,
                    );
                    keypoint_score_map[y * model_width + x] =
                        exponentials[local_y * 8 + local_x] / sum;
                    reliability_score_map[y * model_width + x] = reliability_score;
                }
            }
        }
    }

    let mut candidates = Vec::new();
    let radius = 2usize;
    for y in radius..model_height - radius {
        for x in radius..model_width - radius {
            let keypoint_score = keypoint_score_map[y * model_width + x];
            if keypoint_score <= 0.05 {
                continue;
            }
            let is_maximum = (y - radius..=y + radius).all(|ny| {
                (x - radius..=x + radius).all(|nx| {
                    nx == x && ny == y
                        || keypoint_score_map[ny * model_width + nx] <= keypoint_score
                })
            });
            if !is_maximum {
                continue;
            }
            let source_x = (x as f32 - letterbox.offset_x as f32) / letterbox.scale;
            let source_y = (y as f32 - letterbox.offset_y as f32) / letterbox.scale;
            if source_x >= 0.0
                && source_x < letterbox.source_width as f32
                && source_y >= 0.0
                && source_y < letterbox.source_height as f32
            {
                let score = keypoint_score * reliability_score_map[y * model_width + x];
                candidates.push((x, y, source_x, source_y, score));
            }
        }
    }
    candidates.sort_by(|a, b| b.4.total_cmp(&a.4));

    let mut features = Vec::with_capacity(limit.min(candidates.len()));
    for (x, y, source_x, source_y, score) in candidates.into_iter().take(limit) {
        let feature_x = x as f32 * descriptor_layout.width as f32 / (model_width - 1) as f32 - 0.5;
        let feature_y =
            y as f32 * descriptor_layout.height as f32 / (model_height - 1) as f32 - 0.5;
        let mut descriptor = vec![0.0f32; DESCRIPTOR_DIM];
        let mut norm_sq = 0.0f32;
        for (channel, value) in descriptor.iter_mut().enumerate() {
            *value = bicubic_channel(
                descriptors,
                descriptor_layout,
                feature_x,
                feature_y,
                channel,
            );
            norm_sq += *value * *value;
        }
        if !norm_sq.is_finite() || norm_sq <= 1e-12 {
            continue;
        }
        let inverse_norm = norm_sq.sqrt().recip();
        for value in &mut descriptor {
            *value *= inverse_norm;
        }
        features.push(SparseFeature {
            x: source_x,
            y: source_y,
            confidence: score,
            descriptor,
        });
    }
    Ok(features)
}

fn bilinear_channel(data: &[f32], layout: TensorLayout, x: f32, y: f32, channel: usize) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let sample = |sample_x: i32, sample_y: i32| {
        let sample_x = sample_x.clamp(0, layout.width as i32 - 1) as usize;
        let sample_y = sample_y.clamp(0, layout.height as i32 - 1) as usize;
        data[layout.index(sample_y, sample_x, channel)]
    };
    sample(x0, y0) * (1.0 - dx) * (1.0 - dy)
        + sample(x1, y0) * dx * (1.0 - dy)
        + sample(x0, y1) * (1.0 - dx) * dy
        + sample(x1, y1) * dx * dy
}

fn bicubic_channel(data: &[f32], layout: TensorLayout, x: f32, y: f32, channel: usize) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x_weights = cubic_weights(x - x0 as f32);
    let y_weights = cubic_weights(y - y0 as f32);
    let mut value = 0.0f32;
    for (row, y_weight) in (-1..=2).zip(y_weights) {
        let sample_y = (y0 + row).clamp(0, layout.height as i32 - 1) as usize;
        for (col, x_weight) in (-1..=2).zip(x_weights) {
            let sample_x = (x0 + col).clamp(0, layout.width as i32 - 1) as usize;
            value += data[layout.index(sample_y, sample_x, channel)] * x_weight * y_weight;
        }
    }
    value
}

fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        a * (t3 - 2.0 * t2 + t),
        (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
        -(a + 2.0) * t3 + (2.0 * a + 3.0) * t2 - a * t,
        a * (-t3 + t2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn tensor_layout_accepts_nchw_and_nhwc() {
        let nchw = TensorLayout::parse(&[1, 64, 80, 80], 64, "desc").unwrap();
        let nhwc = TensorLayout::parse(&[1, 80, 80, 64], 64, "desc").unwrap();
        assert!(nchw.channels_first);
        assert!(!nhwc.channels_first);
        assert_eq!(nchw.expected_len(), nhwc.expected_len());
    }

    #[test]
    fn bicubic_sampling_preserves_grid_values_at_integer_coordinates() {
        let layout = TensorLayout::parse(&[1, 1, 4, 4], 1, "values").unwrap();
        let values: Vec<_> = (0..16).map(|value| value as f32).collect();
        assert!((bicubic_channel(&values, layout, 2.0, 1.0, 0) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_handles_images_larger_than_model() {
        let image =
            Mat::new_rows_cols_with_default(720, 1280, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        let (padded, transform) = letterbox_gray(&image, 640, 640).unwrap();
        assert_eq!((padded.cols(), padded.rows()), (640, 640));
        assert!(transform.scale < 1.0);
    }

    #[test]
    fn constant_image_normalization_is_finite() {
        let image =
            Mat::new_rows_cols_with_default(8, 8, core::CV_8UC1, Scalar::all(12.0)).unwrap();
        let values = normalize_gray(&image).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn nms_uses_keypoint_heatmap_before_reliability_ranking() {
        let descriptor_layout = TensorLayout::parse(&[1, 64, 2, 2], 64, "desc").unwrap();
        let keypoint_layout = TensorLayout::parse(&[1, 65, 2, 2], 65, "keypoint").unwrap();
        let mut descriptors = vec![0.0f32; descriptor_layout.expected_len()];
        for y in 0..2 {
            for x in 0..2 {
                descriptors[descriptor_layout.index(y, x, 0)] = 1.0;
            }
        }

        let mut keypoints = vec![0.0f32; keypoint_layout.expected_len()];
        keypoints[keypoint_layout.index(0, 0, 63)] = 6.36;
        keypoints[keypoint_layout.index(0, 1, 56)] = 5.55;
        let reliability = vec![0.01, 1.0, 0.01, 1.0];
        let features = extract_sparse_features(
            &[1, 64, 2, 2],
            &descriptors,
            &[1, 65, 2, 2],
            &keypoints,
            &[1, 1, 2, 2],
            &reliability,
            16,
            16,
            Letterbox {
                scale: 1.0,
                offset_x: 0,
                offset_y: 0,
                source_width: 16,
                source_height: 16,
            },
            1,
        )
        .unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!((features[0].x, features[0].y), (7.0, 7.0));
    }

    #[test]
    fn real_model_runs_without_shape_assumptions() {
        let _guard = crate::MODEL_TEST_LOCK.lock().unwrap();
        let model: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "..",
            "model",
            "xfeat_640x640.onnx",
        ]
        .iter()
        .collect();
        if !model.exists() {
            eprintln!("skipping: xfeat model not present");
            return;
        }
        // The session registers the CUDA EP first; on CPU-only hosts (e.g. CI)
        // that registration fails, so skip rather than panic — mirroring the
        // PIDNet real-inference test's graceful-skip policy.
        // onnxruntime's `load-dynamic` backend panics (rather than returning Err)
        // when the shared library cannot be dlopen'd, so catch that here too.
        let session =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| XfeatExtractor::new(model)));
        let engine = match session {
            Ok(Ok(engine)) => engine,
            Ok(Err(error)) => {
                eprintln!("skipping: XFeat session unavailable here: {error}");
                return;
            }
            Err(_) => {
                eprintln!("skipping: XFeat ONNX Runtime dylib unavailable here");
                return;
            }
        };
        let image =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        let features = engine.extract_features(&image, 32).unwrap();
        assert!(features.len() <= 32);
        assert!(features
            .iter()
            .all(|feature| feature.descriptor.len() == 64));
    }
}
