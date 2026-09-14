use super::{ipm_projector::CameraCalibration, xfeat_engine::SparseFeature};
use opencv::{
    calib3d,
    core::{self, Mat, Point2f},
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::sync::Mutex;

const DESCRIPTOR_DIM: usize = 64;

pub struct SubpixelRefiner {
    inference_session: Option<Mutex<Session>>,
}

impl SubpixelRefiner {
    pub fn new() -> Self {
        let model_path = "model/refinement_mlp.onnx";
        if !std::path::Path::new(model_path).exists() {
            return Self {
                inference_session: None,
            };
        }
        let session = Session::builder()
            .and_then(|builder| {
                builder.with_execution_providers([
                    ort::ep::CUDA::default().build(),
                    ort::ep::CPU::default().build(),
                ])
            })
            .and_then(|builder| builder.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(model_path));
        Self {
            inference_session: session.ok().map(Mutex::new),
        }
    }

    pub fn predict_subpixel_offset(&self, f_a: &[f32], f_b: &[f32]) -> Result<(f32, f32), String> {
        validate_descriptor(f_a)?;
        validate_descriptor(f_b)?;
        let session = self
            .inference_session
            .as_ref()
            .ok_or_else(|| "Subpixel MLP weights unavailable".to_string())?;
        let mut concatenated = Vec::with_capacity(DESCRIPTOR_DIM * 2);
        concatenated.extend_from_slice(f_a);
        concatenated.extend_from_slice(f_b);
        let input = Value::from_array(([1usize, DESCRIPTOR_DIM * 2], concatenated))
            .map_err(|e| e.to_string())?;
        let mut session = session.lock().map_err(|e| e.to_string())?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| e.to_string())?;
        if outputs.len() != 1 {
            return Err(format!(
                "Subpixel MLP expects one output, got {}",
                outputs.len()
            ));
        }
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| "Subpixel MLP output is empty".to_string())?;
        let (shape, logits) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if shape.num_elements() != 64 || logits.len() != 64 || logits.iter().any(|v| !v.is_finite())
        {
            return Err(format!(
                "Subpixel MLP output contract error: shape={shape:?}"
            ));
        }
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "Subpixel MLP output is empty".to_string())?;
        Ok(((best % 8) as f32 - 3.5, (best / 8) as f32 - 3.5))
    }

    /// 在 HWC 64 维稠密描述子图上进行局部抛物线细化,输出原图像素偏移。
    pub fn interpolate_subpixel_offset(
        &self,
        f_a: &[f32],
        desc_tensor2: &[f32],
        w_d8: usize,
        h_d8: usize,
        x2: f32,
        y2: f32,
    ) -> Result<(f32, f32), String> {
        validate_descriptor(f_a)?;
        if w_d8 < 2
            || h_d8 < 2
            || desc_tensor2.len() != w_d8 * h_d8 * DESCRIPTOR_DIM
            || desc_tensor2.iter().any(|v| !v.is_finite())
            || !x2.is_finite()
            || !y2.is_finite()
        {
            return Err("Dense descriptor or image coordinate contract invalid".to_string());
        }
        let image_width = w_d8 * 8;
        let image_height = h_d8 * 8;
        let x_coarse = x2 * w_d8 as f32 / (image_width - 1) as f32 - 0.5;
        let y_coarse = y2 * h_d8 as f32 / (image_height - 1) as f32 - 0.5;
        let step = 0.125f32;
        let score = |x: f32, y: f32| {
            let sampled = interpolate_descriptor(desc_tensor2, w_d8, h_d8, x, y);
            cosine_similarity(f_a, &sampled)
        };
        let center = score(x_coarse, y_coarse);
        let x_minus = score(x_coarse - step, y_coarse);
        let x_plus = score(x_coarse + step, y_coarse);
        let y_minus = score(x_coarse, y_coarse - step);
        let y_plus = score(x_coarse, y_coarse + step);
        let refine = |minus: f32, center: f32, plus: f32| {
            let denominator = minus - 2.0 * center + plus;
            if denominator.abs() > 1e-6 {
                ((minus - plus) / (2.0 * denominator)).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };
        Ok((
            refine(x_minus, center, x_plus),
            refine(y_minus, center, y_plus),
        ))
    }
}

impl Default for SubpixelRefiner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct HomographyEstimate {
    /// 从实时图像像素映射到历史图像像素的 3x3 单应矩阵。
    pub realtime_to_history: [[f64; 3]; 3],
    pub inlier_matches: Vec<(usize, usize, f32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarMetricTransform {
    /// 将实时相机地面坐标变换到历史相机地面坐标。
    pub forward_translation_m: f32,
    pub left_translation_m: f32,
    pub yaw_rad: f32,
    pub inlier_count: usize,
}

pub struct BiomimeticMatcher;

impl BiomimeticMatcher {
    pub fn cross_match(
        realtime_features: &[SparseFeature],
        history_snapshot: &[SparseFeature],
        min_similarity_threshold: f32,
    ) -> Vec<(usize, usize, f32)> {
        if !min_similarity_threshold.is_finite() {
            return Vec::new();
        }
        let valid_realtime: Vec<_> = realtime_features
            .iter()
            .enumerate()
            .filter(|(_, feature)| valid_feature(feature))
            .collect();
        let valid_history: Vec<_> = history_snapshot
            .iter()
            .enumerate()
            .filter(|(_, feature)| valid_feature(feature))
            .collect();
        if valid_realtime.is_empty() || valid_history.is_empty() {
            return Vec::new();
        }

        let forward: Vec<_> = valid_realtime
            .iter()
            .map(|(_, current)| {
                valid_history
                    .iter()
                    .enumerate()
                    .map(|(position, (_, history))| {
                        (
                            position,
                            cosine_similarity(&current.descriptor, &history.descriptor),
                        )
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((usize::MAX, f32::NEG_INFINITY))
            })
            .collect();
        let backward: Vec<_> = valid_history
            .iter()
            .map(|(_, history)| {
                valid_realtime
                    .iter()
                    .enumerate()
                    .map(|(position, (_, current))| {
                        (
                            position,
                            cosine_similarity(&current.descriptor, &history.descriptor),
                        )
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(usize::MAX, |best| best.0)
            })
            .collect();

        forward
            .into_iter()
            .enumerate()
            .filter_map(|(current_position, (history_position, score))| {
                (history_position < backward.len()
                    && backward[history_position] == current_position
                    && score >= min_similarity_threshold)
                    .then_some((
                        valid_realtime[current_position].0,
                        valid_history[history_position].0,
                        score,
                    ))
            })
            .collect()
    }

    pub fn geometry_correction_filter(
        realtime_features: &[SparseFeature],
        history_snapshot: &[SparseFeature],
        matches: &[(usize, usize, f32)],
        ransac_error_threshold: f64,
    ) -> Result<Vec<(usize, usize, f32)>, String> {
        if matches.len() < 8 {
            return Err("Fundamental matrix RANSAC requires at least 8 matches".to_string());
        }
        let (realtime, history) = matched_points(realtime_features, history_snapshot, matches)?;
        let mut mask = Mat::default();
        let fundamental = calib3d::find_fundamental_mat(
            &realtime,
            &history,
            calib3d::FM_RANSAC,
            valid_threshold(ransac_error_threshold)?,
            0.999,
            2000,
            &mut mask,
        )
        .map_err(|e| e.to_string())?;
        if fundamental.empty() || mask.total() != matches.len() {
            return Err(
                "Fundamental matrix estimation degenerate or inlier mask invalid".to_string(),
            );
        }
        filter_by_mask(matches, &mask)
    }

    pub fn estimate_homography(
        realtime_features: &[SparseFeature],
        history_snapshot: &[SparseFeature],
        matches: &[(usize, usize, f32)],
        ransac_error_threshold: f64,
    ) -> Result<HomographyEstimate, String> {
        if matches.len() < 4 {
            return Err("Homography RANSAC requires at least 4 matches".to_string());
        }
        let (realtime, history) = matched_points(realtime_features, history_snapshot, matches)?;
        let mut mask = Mat::default();
        let homography = calib3d::find_homography_ext(
            &realtime,
            &history,
            calib3d::RANSAC,
            valid_threshold(ransac_error_threshold)?,
            &mut mask,
            2000,
            0.999,
        )
        .map_err(|e| e.to_string())?;
        if homography.empty() || homography.rows() != 3 || homography.cols() != 3 {
            return Err("Homography estimation degenerate".to_string());
        }
        let inliers = filter_by_mask(matches, &mask)?;
        if inliers.len() < 4 {
            return Err("Fewer than 4 valid homography inliers".to_string());
        }
        let mut matrix = [[0.0f64; 3]; 3];
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            for (col, value) in matrix_row.iter_mut().enumerate() {
                *value = *homography
                    .at_2d::<f64>(row as i32, col as i32)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(HomographyEstimate {
            realtime_to_history: matrix,
            inlier_matches: inliers,
        })
    }

    /// 仅适用于已知落在地平面上的特征。非地面特征必须使用完整位姿估计。
    pub fn estimate_ground_metric_transform(
        realtime_features: &[SparseFeature],
        history_snapshot: &[SparseFeature],
        matches: &[(usize, usize, f32)],
        calibration: CameraCalibration,
        ransac_error_threshold: f64,
    ) -> Result<PlanarMetricTransform, String> {
        let estimate = Self::estimate_homography(
            realtime_features,
            history_snapshot,
            matches,
            ransac_error_threshold,
        )?;
        let metric_pairs: Vec<_> = estimate
            .inlier_matches
            .iter()
            .filter_map(|(current, history, _)| {
                let current = &realtime_features[*current];
                let history = &history_snapshot[*history];
                Some((
                    calibration.pixel_to_ground(current.x, current.y)?,
                    calibration.pixel_to_ground(history.x, history.y)?,
                ))
            })
            .collect();
        if metric_pairs.len() < 3 {
            return Err("Fewer than 3 valid ground-plane metric matches".to_string());
        }
        let count = metric_pairs.len() as f32;
        let current_center = metric_pairs.iter().fold((0.0, 0.0), |sum, pair| {
            (sum.0 + pair.0 .0, sum.1 + pair.0 .1)
        });
        let history_center = metric_pairs.iter().fold((0.0, 0.0), |sum, pair| {
            (sum.0 + pair.1 .0, sum.1 + pair.1 .1)
        });
        let current_center = (current_center.0 / count, current_center.1 / count);
        let history_center = (history_center.0 / count, history_center.1 / count);
        let (dot, cross) = metric_pairs.iter().fold((0.0, 0.0), |sum, pair| {
            let current = (pair.0 .0 - current_center.0, pair.0 .1 - current_center.1);
            let history = (pair.1 .0 - history_center.0, pair.1 .1 - history_center.1);
            (
                sum.0 + current.0 * history.0 + current.1 * history.1,
                sum.1 + current.0 * history.1 - current.1 * history.0,
            )
        });
        if dot.abs() + cross.abs() <= 1e-8 {
            return Err("Ground-plane metric matches geometrically degenerate".to_string());
        }
        let yaw = cross.atan2(dot);
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let rotated_center = (
            cos_yaw * current_center.0 - sin_yaw * current_center.1,
            sin_yaw * current_center.0 + cos_yaw * current_center.1,
        );
        Ok(PlanarMetricTransform {
            forward_translation_m: history_center.0 - rotated_center.0,
            left_translation_m: history_center.1 - rotated_center.1,
            yaw_rad: yaw,
            inlier_count: metric_pairs.len(),
        })
    }
}

fn valid_feature(feature: &SparseFeature) -> bool {
    feature.x.is_finite()
        && feature.y.is_finite()
        && feature.confidence.is_finite()
        && validate_descriptor(&feature.descriptor).is_ok()
}

fn validate_descriptor(descriptor: &[f32]) -> Result<(), String> {
    if descriptor.len() != DESCRIPTOR_DIM || descriptor.iter().any(|value| !value.is_finite()) {
        return Err("XFeat descriptor must be 64 finite floats".to_string());
    }
    Ok(())
}

fn valid_threshold(threshold: f64) -> Result<f64, String> {
    if threshold.is_finite() && threshold > 0.0 {
        Ok(threshold)
    } else {
        Err("RANSAC threshold must be a finite positive number".to_string())
    }
}

fn matched_points(
    realtime: &[SparseFeature],
    history: &[SparseFeature],
    matches: &[(usize, usize, f32)],
) -> Result<(core::Vector<Point2f>, core::Vector<Point2f>), String> {
    let mut realtime_points = core::Vector::new();
    let mut history_points = core::Vector::new();
    for &(current_index, history_index, score) in matches {
        let current = realtime
            .get(current_index)
            .ok_or_else(|| format!("Realtime feature index out of bounds: {current_index}"))?;
        let historical = history
            .get(history_index)
            .ok_or_else(|| format!("History feature index out of bounds: {history_index}"))?;
        if !valid_feature(current) || !valid_feature(historical) || !score.is_finite() {
            return Err("Match pair contains invalid feature or score".to_string());
        }
        realtime_points.push(Point2f::new(current.x, current.y));
        history_points.push(Point2f::new(historical.x, historical.y));
    }
    Ok((realtime_points, history_points))
}

fn filter_by_mask(
    matches: &[(usize, usize, f32)],
    mask: &Mat,
) -> Result<Vec<(usize, usize, f32)>, String> {
    if mask.total() != matches.len() {
        return Err("RANSAC inlier mask length mismatch".to_string());
    }
    let mut inliers = Vec::new();
    for (index, item) in matches.iter().enumerate() {
        if *mask.at::<u8>(index as i32).map_err(|e| e.to_string())? != 0 {
            inliers.push(*item);
        }
    }
    Ok(inliers)
}

fn interpolate_descriptor(
    tensor: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> [f32; DESCRIPTOR_DIM] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let offset = |sample_x: i32, sample_y: i32| {
        let sample_x = sample_x.clamp(0, width as i32 - 1) as usize;
        let sample_y = sample_y.clamp(0, height as i32 - 1) as usize;
        (sample_y * width + sample_x) * DESCRIPTOR_DIM
    };
    let offsets = [
        offset(x0, y0),
        offset(x0 + 1, y0),
        offset(x0, y0 + 1),
        offset(x0 + 1, y0 + 1),
    ];
    let weights = [
        (1.0 - dx) * (1.0 - dy),
        dx * (1.0 - dy),
        (1.0 - dx) * dy,
        dx * dy,
    ];
    let mut result = [0.0f32; DESCRIPTOR_DIM];
    let mut norm_sq = 0.0f32;
    for channel in 0..DESCRIPTOR_DIM {
        result[channel] = offsets
            .iter()
            .zip(weights)
            .map(|(offset, weight)| tensor[*offset + channel] * weight)
            .sum();
        norm_sq += result[channel] * result[channel];
    }
    let inverse_norm = norm_sq.sqrt().max(1e-12).recip();
    for value in &mut result {
        *value *= inverse_norm;
    }
    result
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let (dot, left_norm_sq, right_norm_sq) =
        left.iter()
            .zip(right)
            .fold((0.0f32, 0.0f32, 0.0f32), |sum, (left, right)| {
                (
                    sum.0 + left * right,
                    sum.1 + left * left,
                    sum.2 + right * right,
                )
            });
    let norm_product = (left_norm_sq * right_norm_sq).sqrt();
    if !norm_product.is_finite() || norm_product <= 1e-12 {
        f32::NEG_INFINITY
    } else {
        (dot / norm_product).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(x: f32, y: f32, descriptor_index: usize) -> SparseFeature {
        let mut descriptor = vec![0.0; 64];
        descriptor[descriptor_index] = 1.0;
        SparseFeature {
            x,
            y,
            confidence: 1.0,
            descriptor,
        }
    }

    #[test]
    fn invalid_descriptors_are_not_matched() {
        let mut invalid = feature(0.0, 0.0, 0);
        invalid.descriptor.pop();
        assert!(
            BiomimeticMatcher::cross_match(&[invalid], &[feature(0.0, 0.0, 0)], 0.5).is_empty()
        );
    }

    #[test]
    fn matching_uses_cosine_similarity_for_scaled_descriptors() {
        let mut realtime = feature(0.0, 0.0, 0);
        let mut history = feature(1.0, 1.0, 0);
        realtime.descriptor[0] = 2.0;
        history.descriptor[0] = 3.0;

        let matches = BiomimeticMatcher::cross_match(&[realtime], &[history], 0.99);
        assert_eq!(matches.len(), 1);
        assert!((matches[0].2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn homography_estimation_returns_inliers() {
        let realtime = vec![
            feature(10.0, 10.0, 0),
            feature(100.0, 10.0, 1),
            feature(10.0, 100.0, 2),
            feature(100.0, 100.0, 3),
            feature(55.0, 55.0, 4),
        ];
        let history: Vec<_> = realtime
            .iter()
            .enumerate()
            .map(|(index, point)| feature(point.x + 4.0, point.y - 2.0, index))
            .collect();
        let matches: Vec<_> = (0..realtime.len())
            .map(|index| (index, index, 1.0))
            .collect();
        let estimate =
            BiomimeticMatcher::estimate_homography(&realtime, &history, &matches, 1.0).unwrap();
        assert_eq!(estimate.inlier_matches.len(), 5);
        assert!((estimate.realtime_to_history[0][2] - 4.0).abs() < 1e-3);
        assert!((estimate.realtime_to_history[1][2] + 2.0).abs() < 1e-3);
    }

    #[test]
    fn ground_matches_produce_metric_translation() {
        let calibration = CameraCalibration {
            image_width: 640,
            image_height: 480,
            fx: 204.25533,
            fy: 153.1915,
            cx: 319.5,
            cy: 239.5,
            forward_offset_m: 0.069,
            left_offset_m: 0.0,
            height_m: 0.133,
            yaw_rad: 0.0,
            pitch_rad: 0.169,
            roll_rad: 0.0,
        };
        let ground_points = [
            (0.8, -0.2),
            (0.8, 0.2),
            (1.2, -0.3),
            (1.2, 0.3),
            (1.8, -0.4),
            (1.8, 0.4),
        ];
        let realtime: Vec<_> = ground_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let pixel = calibration.ground_to_pixel(point.0, point.1).unwrap();
                feature(pixel.0, pixel.1, index)
            })
            .collect();
        let history: Vec<_> = ground_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let pixel = calibration
                    .ground_to_pixel(point.0 + 0.10, point.1 + 0.05)
                    .unwrap();
                feature(pixel.0, pixel.1, index)
            })
            .collect();
        let matches: Vec<_> = (0..ground_points.len())
            .map(|index| (index, index, 1.0))
            .collect();
        let transform = BiomimeticMatcher::estimate_ground_metric_transform(
            &realtime,
            &history,
            &matches,
            calibration,
            1.0,
        )
        .unwrap();
        assert!((transform.forward_translation_m - 0.10).abs() < 1e-3);
        assert!((transform.left_translation_m - 0.05).abs() < 1e-3);
        assert!(transform.yaw_rad.abs() < 1e-3);
    }

    #[test]
    fn subpixel_fallback_rejects_wrong_tensor() {
        let microscope = SubpixelRefiner::new();
        assert!(microscope
            .interpolate_subpixel_offset(&[0.0; 64], &[0.0; 63], 1, 1, 0.0, 0.0)
            .is_err());
    }
}
