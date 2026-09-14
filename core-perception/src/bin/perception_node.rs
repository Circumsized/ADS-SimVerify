use core_perception::perception::{
    ipm_projector::{
        BevConfig, BevProjection, CameraCalibration, IpmProjector, BEV_SEMANTIC_CHANNELS, OCCUPIED,
    },
    pidnet_engine::{PidnetEngine, PIDNET_MIN_CONFIDENCE, PIDNET_MODEL_HEIGHT, PIDNET_MODEL_WIDTH},
    xfeat_engine::{SparseFeature, XfeatExtractor},
};
use dora_node_api::{
    arrow::{
        array::{ArrayRef, FixedSizeListArray, Float32Array, StructArray, UInt8Array},
        datatypes::{DataType, Field},
    },
    DoraNode, Event, Metadata, MetadataParameters, Parameter,
};
use eyre::{eyre, Context};
use opencv::prelude::*;
use std::{
    env,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

const BEV_WIDTH: i32 = 192;
const BEV_HEIGHT: i32 = 192;
const BEV_PHYSICAL_SIZE_M: f32 = 20.0;
const TARGET_PERIOD_MS: u64 = 50;
const MAX_INPUT_AGE: Duration = Duration::from_millis(200);
const MAX_FUTURE_SKEW: Duration = Duration::from_millis(50);

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let calibration = calibration_from_env()?;
    let bev_config = BevConfig {
        width: BEV_WIDTH,
        height: BEV_HEIGHT,
        meters_per_cell: required_env_f32("BEV_METERS_PER_CELL")?,
        min_forward_m: required_env_f32("BEV_MIN_FORWARD_M")?,
        ego_row: required_env_f32("BEV_EGO_ROW")?,
        ego_col: required_env_f32("BEV_EGO_COL")?,
    };
    validate_deployment_geometry(calibration, bev_config)?;
    let pidnet_path =
        env::var("PIDNET_MODEL_PATH").unwrap_or_else(|_| "model/pidnet_s.onnx".to_string());
    let xfeat_path =
        env::var("XFEAT_MODEL_PATH").unwrap_or_else(|_| "model/xfeat_640x640.onnx".to_string());
    let xfeat_interval = env::var("XFEAT_FRAME_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10);

    let pidnet =
        PidnetEngine::new(&pidnet_path).map_err(|e| eyre!("PIDNet initialization failed: {e}"))?;
    let projector = IpmProjector::new(calibration, bev_config)
        .map_err(|e| eyre!("IPM initialization failed: {e}"))?;
    let xfeat = match XfeatExtractor::new(&xfeat_path) {
        Ok(engine) => Some(engine),
        Err(error) => {
            eprintln!("[perception] XFeat initialization failed, will only publish segmentation results: {error}");
            None
        }
    };

    eprintln!(
        "[perception] ready: camera={}x{}, BEV={}x{} @ {:.5}m/cell, {:.2}x{:.2}m",
        calibration.image_width,
        calibration.image_height,
        bev_config.width,
        bev_config.height,
        bev_config.meters_per_cell,
        bev_config.physical_width_m(),
        bev_config.physical_forward_m()
    );

    let mut sequence = 0u64;
    let mut last_started = None::<Instant>;
    while let Some(event) = events.recv_async().await {
        match event {
            Event::Input { id, metadata, data } if id.as_str() == "jpeg_image" => {
                if last_started
                    .is_some_and(|last| last.elapsed().as_millis() < TARGET_PERIOD_MS as u128)
                {
                    continue;
                }
                last_started = Some(Instant::now());
                sequence += 1;
                let started = Instant::now();
                if let Err(error) = validate_source_freshness(&metadata) {
                    publish_fail_safe(&mut node, &metadata, sequence, started, bev_config, &error)?;
                    continue;
                }

                let frame = dora_node_api::into_vec::<u8>(&data)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| decode_jpeg(&bytes));
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(error) => {
                        publish_fail_safe(
                            &mut node,
                            &metadata,
                            sequence,
                            started,
                            bev_config,
                            &format!("JPEG decode failed: {error}"),
                        )?;
                        continue;
                    }
                };
                if let Err(error) = validate_frame_geometry(&frame, calibration) {
                    publish_fail_safe(&mut node, &metadata, sequence, started, bev_config, &error)?;
                    continue;
                }

                let projection = pidnet
                    .segment(&frame)
                    .and_then(|class_map| projector.project(&class_map));
                match projection {
                    Ok(projection) => publish_projection(
                        &mut node, &metadata, sequence, started, bev_config, projection,
                    )?,
                    Err(error) => publish_fail_safe(
                        &mut node,
                        &metadata,
                        sequence,
                        started,
                        bev_config,
                        &format!("Segmentation/IPM failed: {error}"),
                    )?,
                }

                if sequence.is_multiple_of(xfeat_interval) {
                    if let Some(engine) = &xfeat {
                        match engine.extract_features(&frame, 200) {
                            Ok(features) => {
                                publish_features(&mut node, &metadata, sequence, started, features)?
                            }
                            Err(error) => {
                                eprintln!("[perception] XFeat frame {sequence} failed: {error}")
                            }
                        }
                    }
                }
            }
            Event::InputClosed { id } if id.as_str() == "jpeg_image" => {
                eprintln!("[perception] jpeg_image input closed, exiting node");
                break;
            }
            Event::Stop(_) => break,
            Event::Error(error) => eprintln!("[perception] Dora error: {error}"),
            _ => {}
        }
    }
    Ok(())
}

fn calibration_from_env() -> eyre::Result<CameraCalibration> {
    Ok(CameraCalibration {
        image_width: required_env_i32("CAMERA_IMAGE_WIDTH")?,
        image_height: required_env_i32("CAMERA_IMAGE_HEIGHT")?,
        fx: required_env_f32("CAMERA_FX")?,
        fy: required_env_f32("CAMERA_FY")?,
        cx: required_env_f32("CAMERA_CX")?,
        cy: required_env_f32("CAMERA_CY")?,
        forward_offset_m: required_env_f32("CAMERA_FORWARD_OFFSET_M")?,
        left_offset_m: required_env_f32("CAMERA_LEFT_OFFSET_M")?,
        height_m: required_env_f32("CAMERA_HEIGHT_M")?,
        yaw_rad: required_env_f32("CAMERA_YAW_RAD")?,
        pitch_rad: required_env_f32("CAMERA_PITCH_RAD")?,
        roll_rad: required_env_f32("CAMERA_ROLL_RAD")?,
    })
}

fn required_env_f32(name: &str) -> eyre::Result<f32> {
    env::var(name)
        .wrap_err_with(|| format!("Missing required calibration parameter {name}"))?
        .parse::<f32>()
        .wrap_err_with(|| format!("Calibration parameter {name} is not a valid f32"))
}

fn required_env_i32(name: &str) -> eyre::Result<i32> {
    env::var(name)
        .wrap_err_with(|| format!("Missing required calibration parameter {name}"))?
        .parse::<i32>()
        .wrap_err_with(|| format!("Calibration parameter {name} is not a valid i32"))
}

fn validate_deployment_geometry(
    calibration: CameraCalibration,
    config: BevConfig,
) -> eyre::Result<()> {
    if (calibration.image_width, calibration.image_height)
        != (PIDNET_MODEL_WIDTH, PIDNET_MODEL_HEIGHT)
    {
        return Err(eyre!(
            "Camera calibration size must match PIDNet output {}x{}, got {}x{}",
            PIDNET_MODEL_WIDTH,
            PIDNET_MODEL_HEIGHT,
            calibration.image_width,
            calibration.image_height
        ));
    }
    let expected_resolution = BEV_PHYSICAL_SIZE_M / BEV_WIDTH as f32;
    let expected_ego_row = (BEV_HEIGHT as f32 - 1.0) * 0.5;
    let expected_ego_col = (BEV_WIDTH as f32 - 1.0) * 0.5;
    if (config.meters_per_cell - expected_resolution).abs() > 1e-6
        || (config.ego_row - expected_ego_row).abs() > 1e-6
        || (config.ego_col - expected_ego_col).abs() > 1e-6
    {
        return Err(eyre!(
            "BEV contract must be {}x{}, {:.9}m/cell, ego=({:.1},{:.1})",
            BEV_WIDTH,
            BEV_HEIGHT,
            expected_resolution,
            expected_ego_row,
            expected_ego_col
        ));
    }
    Ok(())
}

fn validate_frame_geometry(frame: &Mat, calibration: CameraCalibration) -> Result<(), String> {
    if (frame.cols(), frame.rows()) != (calibration.image_width, calibration.image_height) {
        return Err(format!(
            "Image size does not match camera calibration: image={}x{} calibration={}x{}",
            frame.cols(),
            frame.rows(),
            calibration.image_width,
            calibration.image_height
        ));
    }
    Ok(())
}

fn decode_jpeg(bytes: &[u8]) -> Result<Mat, String> {
    if bytes.is_empty() {
        return Err("Input bytes are empty".to_string());
    }
    let source = opencv::core::Vector::<u8>::from_slice(bytes);
    let frame = opencv::imgcodecs::imdecode(&source, opencv::imgcodecs::IMREAD_COLOR)
        .map_err(|e| e.to_string())?;
    if frame.empty() {
        Err("OpenCV returned empty image".to_string())
    } else {
        Ok(frame)
    }
}

fn validate_source_freshness(metadata: &Metadata) -> Result<(), String> {
    let source_time = metadata.timestamp().get_time().to_system_time();
    match SystemTime::now().duration_since(source_time) {
        Ok(age) if age <= MAX_INPUT_AGE => Ok(()),
        Ok(age) => Err(format!(
            "Input image expired: age_ms={:.1}",
            age.as_secs_f64() * 1000.0
        )),
        Err(error) if error.duration() <= MAX_FUTURE_SKEW => Ok(()),
        Err(error) => Err(format!(
            "Input image timestamp in the future: skew_ms={:.1}",
            error.duration().as_secs_f64() * 1000.0
        )),
    }
}

fn publish_projection(
    node: &mut DoraNode,
    source: &Metadata,
    sequence: u64,
    started: Instant,
    config: BevConfig,
    projection: BevProjection,
) -> eyre::Result<()> {
    let occupancy = flatten_u8_mat(&projection.occupancy)
        .map_err(|error| eyre!("BEV flatten failed: {error}"))?;
    let metadata = bev_metadata(source, sequence, started, config, true, None);
    node.send_output(
        "bev_grid".to_string().into(),
        metadata.clone(),
        UInt8Array::from(occupancy),
    )?;
    node.send_output(
        "bev_semantic".to_string().into(),
        metadata,
        UInt8Array::from(projection.semantic_chw),
    )?;
    Ok(())
}

fn publish_fail_safe(
    node: &mut DoraNode,
    source: &Metadata,
    sequence: u64,
    started: Instant,
    config: BevConfig,
    reason: &str,
) -> eyre::Result<()> {
    eprintln!("[perception] frame {sequence} fail-safe: {reason}");
    let plane_len = (config.width * config.height) as usize;
    let occupancy = vec![OCCUPIED; plane_len];
    let mut semantics = vec![0u8; plane_len * BEV_SEMANTIC_CHANNELS];
    semantics[(BEV_SEMANTIC_CHANNELS - 1) * plane_len..].fill(1);
    let metadata = bev_metadata(source, sequence, started, config, false, Some(reason));
    node.send_output(
        "bev_grid".to_string().into(),
        metadata.clone(),
        UInt8Array::from(occupancy),
    )?;
    node.send_output(
        "bev_semantic".to_string().into(),
        metadata,
        UInt8Array::from(semantics),
    )?;
    Ok(())
}

fn bev_metadata(
    source: &Metadata,
    sequence: u64,
    started: Instant,
    config: BevConfig,
    valid: bool,
    error: Option<&str>,
) -> MetadataParameters {
    let mut metadata = MetadataParameters::new();
    metadata.insert(
        "source_timestamp".into(),
        Parameter::String(source.timestamp().to_string()),
    );
    if let Ok(age) =
        SystemTime::now().duration_since(source.timestamp().get_time().to_system_time())
    {
        metadata.insert(
            "source_age_ms".into(),
            Parameter::Float(age.as_secs_f64() * 1000.0),
        );
    }
    metadata.insert("sequence".into(), Parameter::Integer(sequence as i64));
    for key in ["source_frame_id", "sim_time_s", "source_kind"] {
        if let Some(value) = source.parameters.get(key) {
            metadata.insert(key.into(), value.clone());
        }
    }
    metadata.insert(
        "shape".into(),
        Parameter::ListInt(vec![config.height as i64, config.width as i64]),
    );
    metadata.insert(
        "semantic_shape".into(),
        Parameter::ListInt(vec![
            BEV_SEMANTIC_CHANNELS as i64,
            config.height as i64,
            config.width as i64,
        ]),
    );
    metadata.insert(
        "frame".into(),
        Parameter::String("E:x-forward,y-left".into()),
    );
    metadata.insert(
        "meters_per_cell".into(),
        Parameter::Float(config.meters_per_cell as f64),
    );
    metadata.insert(
        "occupancy_encoding".into(),
        Parameter::String("0=observed-road,255=occupied-or-unknown".into()),
    );
    metadata.insert(
        "segmentation_classes".into(),
        Parameter::String("cityscapes-train-id-19".into()),
    );
    metadata.insert("segmentation_ignore_label".into(), Parameter::Integer(255));
    metadata.insert(
        "segmentation_min_confidence".into(),
        Parameter::Float(PIDNET_MIN_CONFIDENCE as f64),
    );
    metadata.insert(
        "semantic_layout".into(),
        Parameter::String("CHW-one-hot".into()),
    );
    metadata.insert(
        "semantic_channels".into(),
        Parameter::ListString(
            [
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic-control",
                "vegetation",
                "terrain",
                "person-or-rider",
                "car",
                "heavy-vehicle",
                "two-wheel-vehicle",
                "unknown-or-sky",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        ),
    );
    metadata.insert(
        "ego_origin_cell".into(),
        Parameter::ListFloat(vec![config.ego_row as f64, config.ego_col as f64]),
    );
    metadata.insert("target_frequency_hz".into(), Parameter::Float(20.0));
    metadata.insert("valid".into(), Parameter::Bool(valid));
    metadata.insert(
        "latency_ms".into(),
        Parameter::Float(started.elapsed().as_secs_f64() * 1000.0),
    );
    if let Some(error) = error {
        metadata.insert("error".into(), Parameter::String(error.to_string()));
    }
    metadata
}

fn publish_features(
    node: &mut DoraNode,
    source: &Metadata,
    sequence: u64,
    started: Instant,
    features: Vec<SparseFeature>,
) -> eyre::Result<()> {
    let mut x = Vec::with_capacity(features.len());
    let mut y = Vec::with_capacity(features.len());
    let mut score = Vec::with_capacity(features.len());
    let mut descriptor = Vec::with_capacity(features.len() * 64);
    for feature in features {
        x.push(feature.x);
        y.push(feature.y);
        score.push(feature.confidence);
        descriptor.extend(feature.descriptor);
    }
    let descriptor_values: ArrayRef = Arc::new(Float32Array::from(descriptor));
    let descriptor_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        64,
        descriptor_values,
        None,
    )?;
    let array = StructArray::from(vec![
        (
            Arc::new(Field::new("x", DataType::Float32, false)),
            Arc::new(Float32Array::from(x)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("y", DataType::Float32, false)),
            Arc::new(Float32Array::from(y)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("score", DataType::Float32, false)),
            Arc::new(Float32Array::from(score)) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "descriptor",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 64),
                false,
            )),
            Arc::new(descriptor_array) as ArrayRef,
        ),
    ]);
    let mut metadata = MetadataParameters::new();
    metadata.insert(
        "source_timestamp".into(),
        Parameter::String(source.timestamp().to_string()),
    );
    metadata.insert("sequence".into(), Parameter::Integer(sequence as i64));
    metadata.insert(
        "coordinate_frame".into(),
        Parameter::String("source-image-pixels".into()),
    );
    metadata.insert("descriptor_dim".into(), Parameter::Integer(64));
    metadata.insert(
        "latency_ms".into(),
        Parameter::Float(started.elapsed().as_secs_f64() * 1000.0),
    );
    node.send_output("xfeat_features".to_string().into(), metadata, array)?;
    Ok(())
}

fn flatten_u8_mat(mat: &Mat) -> Result<Vec<u8>, String> {
    if mat.typ() != opencv::core::CV_8UC1 || mat.rows() <= 0 || mat.cols() <= 0 {
        return Err(format!(
            "Expected non-empty CV_8UC1, got {}x{} type={}",
            mat.cols(),
            mat.rows(),
            mat.typ()
        ));
    }
    let mut result = Vec::with_capacity((mat.rows() * mat.cols()) as usize);
    for row in 0..mat.rows() {
        let pointer = mat.ptr(row).map_err(|e| e.to_string())?;
        let slice = unsafe { std::slice::from_raw_parts(pointer, mat.cols() as usize) };
        result.extend_from_slice(slice);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_safe_semantic_shape_matches_contract() {
        let config = BevConfig {
            width: 192,
            height: 192,
            meters_per_cell: 20.0 / 192.0,
            min_forward_m: 0.2,
            ego_row: 95.5,
            ego_col: 95.5,
        };
        assert_eq!(
            config.width as usize * config.height as usize * BEV_SEMANTIC_CHANNELS,
            516_096
        );
    }

    #[test]
    fn deployment_geometry_matches_pidnet_and_control_contracts() {
        let calibration = CameraCalibration {
            image_width: PIDNET_MODEL_WIDTH,
            image_height: PIDNET_MODEL_HEIGHT,
            fx: 200.0,
            fy: 150.0,
            cx: 319.5,
            cy: 239.5,
            forward_offset_m: 0.0,
            left_offset_m: 0.0,
            height_m: 0.2,
            yaw_rad: 0.0,
            pitch_rad: 0.2,
            roll_rad: 0.0,
        };
        let config = BevConfig {
            width: BEV_WIDTH,
            height: BEV_HEIGHT,
            meters_per_cell: BEV_PHYSICAL_SIZE_M / BEV_WIDTH as f32,
            min_forward_m: 0.2,
            ego_row: 95.5,
            ego_col: 95.5,
        };
        assert!(validate_deployment_geometry(calibration, config).is_ok());

        let wrong_frame = Mat::new_rows_cols_with_default(
            720,
            1280,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(0.0),
        )
        .unwrap();
        assert!(validate_frame_geometry(&wrong_frame, calibration).is_err());
    }
}
