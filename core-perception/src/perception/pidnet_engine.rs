use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::sync::Mutex;

use super::ipm_projector::CITYSCAPES_CLASS_COUNT;

const INPUT_NAME: &str = "input";
const OUTPUT_NAME: &str = "output";
const IGNORE_LABEL: u8 = 255;
pub const PIDNET_MIN_CONFIDENCE: f32 = 0.50;
pub const PIDNET_MODEL_WIDTH: i32 = 640;
pub const PIDNET_MODEL_HEIGHT: i32 = 480;

pub struct PidnetEngine {
    session: Mutex<Session>,
    model_width: i32,
    model_height: i32,
}

impl PidnetEngine {
    pub fn new<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self, String> {
        crate::self_heal_load_onnx_dylib();
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_execution_providers([
                ort::ep::CUDA::default().build(),
                ort::ep::CPU::default().build(),
            ])
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(1)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;

        validate_model_names(&session)?;
        Ok(Self {
            session: Mutex::new(session),
            model_width: PIDNET_MODEL_WIDTH,
            model_height: PIDNET_MODEL_HEIGHT,
        })
    }

    pub fn segment(&self, input_image: &Mat) -> Result<Mat, String> {
        validate_input_image(input_image)?;
        let mut resized = Mat::default();
        imgproc::resize(
            input_image,
            &mut resized,
            Size::new(self.model_width, self.model_height),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .map_err(|e| e.to_string())?;

        let mut rgb_image = Mat::default();
        let conversion = match resized.channels() {
            3 => imgproc::COLOR_BGR2RGB,
            4 => imgproc::COLOR_BGRA2RGB,
            channels => return Err(format!("PIDNet 不支持 {channels} 通道输入")),
        };
        imgproc::cvt_color(&resized, &mut rgb_image, conversion, 0).map_err(|e| e.to_string())?;

        let h = self.model_height as usize;
        let w = self.model_width as usize;
        let tensor_data = image_to_nchw(&rgb_image, w, h)?;
        let input_value =
            Value::from_array(([1usize, 3, h, w], tensor_data)).map_err(|e| e.to_string())?;

        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        let outputs = session
            .run(ort::inputs![INPUT_NAME => input_value])
            .map_err(|e| e.to_string())?;
        let output = outputs
            .get(OUTPUT_NAME)
            .ok_or_else(|| format!("PIDNet 缺少命名输出 {OUTPUT_NAME}"))?;
        let (shape, logits) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let small = decode_class_map(shape, logits, PIDNET_MIN_CONFIDENCE)?;

        let mut class_map = Mat::default();
        imgproc::resize(
            &small,
            &mut class_map,
            Size::new(self.model_width, self.model_height),
            0.0,
            0.0,
            imgproc::INTER_NEAREST,
        )
        .map_err(|e| e.to_string())?;
        Ok(class_map)
    }
}

fn validate_model_names(session: &Session) -> Result<(), String> {
    if !session
        .inputs()
        .iter()
        .any(|input| input.name() == INPUT_NAME)
    {
        return Err(format!("PIDNet model missing input {INPUT_NAME}"));
    }
    if !session
        .outputs()
        .iter()
        .any(|output| output.name() == OUTPUT_NAME)
    {
        return Err(format!("PIDNet model missing output {OUTPUT_NAME}"));
    }
    Ok(())
}

fn validate_input_image(image: &Mat) -> Result<(), String> {
    if image.empty() || image.rows() <= 0 || image.cols() <= 0 {
        return Err("PIDNet 输入图像为空".to_string());
    }
    if image.depth() != core::CV_8U {
        return Err(format!(
            "PIDNet only accepts CV_8U images, got depth={}",
            image.depth()
        ));
    }
    if !matches!(image.channels(), 3 | 4) {
        return Err(format!(
            "PIDNet 只接受 BGR/BGRA 图像，实际 channels={}",
            image.channels()
        ));
    }
    Ok(())
}

fn image_to_nchw(rgb: &Mat, width: usize, height: usize) -> Result<Vec<f32>, String> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut tensor = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        let row = rgb.ptr(y as i32).map_err(|e| e.to_string())?;
        let row = unsafe { std::slice::from_raw_parts(row, width * 3) };
        for x in 0..width {
            for channel in 0..3 {
                let value = row[x * 3 + channel] as f32 / 255.0;
                tensor[(channel * height + y) * width + x] = (value - mean[channel]) / std[channel];
            }
        }
    }
    Ok(tensor)
}

fn decode_class_map(shape: &[i64], logits: &[f32], min_confidence: f32) -> Result<Mat, String> {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        return Err("PIDNet min confidence must be in [0, 1]".to_string());
    }
    if shape.len() != 4 || shape[0] != 1 {
        return Err(format!(
            "PIDNet 输出必须是 batch=1 的四维张量，实际 {shape:?}"
        ));
    }
    let class_count = CITYSCAPES_CLASS_COUNT as i64;
    let (channels_first, out_h, out_w) = if shape[1] == class_count {
        (true, shape[2], shape[3])
    } else if shape[3] == class_count {
        (false, shape[1], shape[2])
    } else {
        return Err(format!(
            "PIDNet output must contain {class_count} class channels, got {shape:?}"
        ));
    };
    if out_h <= 0 || out_w <= 0 {
        return Err(format!("PIDNet 输出空间尺寸无效：{shape:?}"));
    }
    let h = out_h as usize;
    let w = out_w as usize;
    let expected = h * w * class_count as usize;
    if logits.len() != expected {
        return Err(format!(
            "PIDNet output element count mismatch: shape={shape:?}, expected {expected}, got {}",
            logits.len()
        ));
    }

    let mut class_map = Mat::new_rows_cols_with_default(
        out_h as i32,
        out_w as i32,
        core::CV_8UC1,
        Scalar::all(IGNORE_LABEL as f64),
    )
    .map_err(|e| e.to_string())?;
    for y in 0..h {
        let row = class_map.ptr_mut(y as i32).map_err(|e| e.to_string())?;
        let row = unsafe { std::slice::from_raw_parts_mut(row, w) };
        for (x, output_class) in row.iter_mut().enumerate().take(w) {
            let mut best = None::<(u8, f32)>;
            for class in 0..class_count as usize {
                let index = if channels_first {
                    (class * h + y) * w + x
                } else {
                    (y * w + x) * class_count as usize + class
                };
                let value = logits[index];
                if value.is_finite() && best.is_none_or(|(_, best_value)| value > best_value) {
                    best = Some((class as u8, value));
                }
            }
            let Some((best_class, best_logit)) = best else {
                *output_class = IGNORE_LABEL;
                continue;
            };
            let mut exponential_sum = 0.0f32;
            for class in 0..class_count as usize {
                let index = if channels_first {
                    (class * h + y) * w + x
                } else {
                    (y * w + x) * class_count as usize + class
                };
                let value = logits[index];
                if value.is_finite() {
                    exponential_sum += (value - best_logit).exp();
                }
            }
            let confidence = exponential_sum.recip();
            *output_class = if confidence.is_finite() && confidence >= min_confidence {
                best_class
            } else {
                IGNORE_LABEL
            };
        }
    }
    Ok(class_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn decodes_nchw_and_marks_all_nan_as_ignore() {
        let mut logits = vec![0.0f32; 19 * 2];
        logits[3 * 2] = 10.0;
        for class in 0..19 {
            logits[class * 2 + 1] = f32::NAN;
        }
        let map = decode_class_map(&[1, 19, 1, 2], &logits, 0.5).unwrap();
        assert_eq!(*map.at_2d::<u8>(0, 0).unwrap(), 3);
        assert_eq!(*map.at_2d::<u8>(0, 1).unwrap(), IGNORE_LABEL);
    }

    #[test]
    fn rejects_wrong_output_shape() {
        assert!(decode_class_map(&[1, 18, 2, 2], &[0.0; 72], 0.5).is_err());
    }

    #[test]
    fn real_model_runs_and_returns_contract_shape() {
        let _guard = crate::MODEL_TEST_LOCK.lock().unwrap();
        let model: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "model", "pidnet_s.onnx"]
            .iter()
            .collect();
        if !model.exists() {
            eprintln!(
                "Skipping inactive PIDNet integration test: {}",
                model.display()
            );
            return;
        }
        let engine = PidnetEngine::new(model).unwrap();
        let image =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        let result = engine.segment(&image).unwrap();
        assert_eq!(
            (result.cols(), result.rows(), result.typ()),
            (640, 480, core::CV_8UC1)
        );
    }
}
