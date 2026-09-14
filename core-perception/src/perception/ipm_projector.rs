use opencv::{
    core::{self, Mat, Scalar},
    imgproc,
    prelude::*,
};

pub const CITYSCAPES_CLASS_COUNT: u8 = 19;
pub const BEV_SEMANTIC_CHANNELS: usize = 14;
pub const OCCUPIED: u8 = 255;
pub const FREE: u8 = 0;

/// 针孔相机标定。车体坐标系为 x 向前、y 向左、z 向上；相机坐标系为
/// x 向右、y 向下、z 向前。pitch 为相机光轴相对车体水平面的向下俯角。
#[derive(Debug, Clone, Copy)]
pub struct CameraCalibration {
    pub image_width: i32,
    pub image_height: i32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub forward_offset_m: f32,
    pub left_offset_m: f32,
    pub height_m: f32,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub roll_rad: f32,
}

impl CameraCalibration {
    fn validate(self) -> Result<Self, String> {
        if self.image_width <= 0 || self.image_height <= 0 {
            return Err("Camera image dimensions must be positive".to_string());
        }
        if !self.fx.is_finite() || !self.fy.is_finite() || self.fx <= 0.0 || self.fy <= 0.0 {
            return Err("Camera focal length must be a finite positive number".to_string());
        }
        if !self.cx.is_finite()
            || !self.cy.is_finite()
            || !self.forward_offset_m.is_finite()
            || !self.left_offset_m.is_finite()
            || !self.height_m.is_finite()
            || self.height_m <= 0.0
            || !self.yaw_rad.is_finite()
            || !self.pitch_rad.is_finite()
            || !self.roll_rad.is_finite()
        {
            return Err("相机主点、高度、pitch 和 roll 必须是有效标定值".to_string());
        }
        Ok(self)
    }

    /// 将图像像素与地平面求交，返回车体坐标系中的 (向前米数, 向左米数)。
    pub fn pixel_to_ground(self, u: f32, v: f32) -> Option<(f32, f32)> {
        if self.validate().is_err() || !u.is_finite() || !v.is_finite() {
            return None;
        }
        let normalized_x = (u - self.cx) / self.fx;
        let normalized_y = (v - self.cy) / self.fy;
        let (sin_pitch, cos_pitch) = self.pitch_rad.sin_cos();
        let (sin_roll, cos_roll) = self.roll_rad.sin_cos();
        let pitched_y = sin_roll * normalized_x + cos_roll * normalized_y;
        let denominator = cos_pitch * pitched_y + sin_pitch;
        if !denominator.is_finite() || denominator <= 1e-6 {
            return None;
        }
        let ray_scale = self.height_m / denominator;
        let heading_forward = ray_scale * (-sin_pitch * pitched_y + cos_pitch);
        let heading_left = -ray_scale * (cos_roll * normalized_x - sin_roll * normalized_y);
        let (sin_yaw, cos_yaw) = self.yaw_rad.sin_cos();
        let forward_m = cos_yaw * heading_forward - sin_yaw * heading_left + self.forward_offset_m;
        let left_m = sin_yaw * heading_forward + cos_yaw * heading_left + self.left_offset_m;
        (forward_m.is_finite() && forward_m >= 0.0 && left_m.is_finite())
            .then_some((forward_m, left_m))
    }

    pub fn ground_to_pixel(self, forward_m: f32, left_m: f32) -> Option<(f32, f32)> {
        if self.validate().is_err() || forward_m < 0.0 || !left_m.is_finite() {
            return None;
        }
        project_ground_point(self, forward_m, left_m)
    }
}

/// 局部车体 BEV：ego_row/ego_col 是车体原点，行减小表示向前，列减小表示向左。
#[derive(Debug, Clone, Copy)]
pub struct BevConfig {
    pub width: i32,
    pub height: i32,
    pub meters_per_cell: f32,
    pub min_forward_m: f32,
    pub ego_row: f32,
    pub ego_col: f32,
}

impl BevConfig {
    pub fn physical_width_m(self) -> f32 {
        self.width as f32 * self.meters_per_cell
    }

    pub fn physical_forward_m(self) -> f32 {
        self.height as f32 * self.meters_per_cell
    }

    fn validate(self) -> Result<Self, String> {
        if self.width <= 0 || self.height <= 0 {
            return Err("BEV dimensions must be positive".to_string());
        }
        if !self.meters_per_cell.is_finite() || self.meters_per_cell <= 0.0 {
            return Err("BEV metric resolution must be a finite positive number".to_string());
        }
        if !self.min_forward_m.is_finite() || self.min_forward_m < 0.0 {
            return Err(
                "BEV near-vehicle blind zone must be a finite non-negative number".to_string(),
            );
        }
        if !self.ego_row.is_finite()
            || !self.ego_col.is_finite()
            || !(0.0..=self.height as f32 - 1.0).contains(&self.ego_row)
            || !(0.0..=self.width as f32 - 1.0).contains(&self.ego_col)
        {
            return Err("BEV ego origin must be inside the grid".to_string());
        }
        Ok(self)
    }
}

pub struct BevProjection {
    /// 与现有控制器兼容的单通道占用图：0 表示已观测道路，255 表示占用或未知。
    pub occupancy: Mat,
    /// CHW 布局的 14 通道二值语义图，每个像素恰好激活一个通道。
    pub semantic_chw: Vec<u8>,
}

pub struct IpmProjector {
    calibration: CameraCalibration,
    config: BevConfig,
    map_x: Mat,
    map_y: Mat,
}

impl IpmProjector {
    pub fn new(calibration: CameraCalibration, config: BevConfig) -> Result<Self, String> {
        let calibration = calibration.validate()?;
        let config = config.validate()?;
        let mut map_x = Mat::new_rows_cols_with_default(
            config.height,
            config.width,
            core::CV_32FC1,
            Scalar::all(-1.0),
        )
        .map_err(|e| e.to_string())?;
        let mut map_y = Mat::new_rows_cols_with_default(
            config.height,
            config.width,
            core::CV_32FC1,
            Scalar::all(-1.0),
        )
        .map_err(|e| e.to_string())?;

        let mut valid_cells = 0usize;
        for row in 0..config.height {
            for col in 0..config.width {
                let forward_m = (config.ego_row - row as f32) * config.meters_per_cell;
                let left_m = (config.ego_col - col as f32) * config.meters_per_cell;
                if forward_m < config.min_forward_m {
                    continue;
                }
                if let Some((u, v)) = project_ground_point(calibration, forward_m, left_m) {
                    if u >= 0.0
                        && u <= (calibration.image_width - 1) as f32
                        && v >= 0.0
                        && v <= (calibration.image_height - 1) as f32
                    {
                        *map_x
                            .at_2d_mut::<f32>(row, col)
                            .map_err(|e| e.to_string())? = u;
                        *map_y
                            .at_2d_mut::<f32>(row, col)
                            .map_err(|e| e.to_string())? = v;
                        valid_cells += 1;
                    }
                }
            }
        }
        if valid_cells == 0 {
            return Err(
                "Camera calibration and BEV range have no visible ground intersection".to_string(),
            );
        }

        Ok(Self {
            calibration,
            config,
            map_x,
            map_y,
        })
    }

    pub fn calibration(&self) -> CameraCalibration {
        self.calibration
    }

    pub fn config(&self) -> BevConfig {
        self.config
    }

    pub fn project(&self, class_map: &Mat) -> Result<BevProjection, String> {
        if class_map.rows() != self.calibration.image_height
            || class_map.cols() != self.calibration.image_width
            || class_map.typ() != core::CV_8UC1
        {
            return Err(format!(
                "Segmentation map contract mismatch: expected {}x{} CV_8UC1, got {}x{} type={}",
                self.calibration.image_width,
                self.calibration.image_height,
                class_map.cols(),
                class_map.rows(),
                class_map.typ()
            ));
        }

        let mut bev_classes = Mat::default();
        imgproc::remap(
            class_map,
            &mut bev_classes,
            &self.map_x,
            &self.map_y,
            imgproc::INTER_NEAREST,
            core::BORDER_CONSTANT,
            Scalar::all(255.0),
        )
        .map_err(|e| e.to_string())?;

        encode_bev(&bev_classes, self.config)
    }
}

fn project_ground_point(
    calibration: CameraCalibration,
    forward_m: f32,
    left_m: f32,
) -> Option<(f32, f32)> {
    let (sin_yaw, cos_yaw) = calibration.yaw_rad.sin_cos();
    let relative_forward = forward_m - calibration.forward_offset_m;
    let relative_left = left_m - calibration.left_offset_m;
    let heading_forward = cos_yaw * relative_forward + sin_yaw * relative_left;
    let heading_left = -sin_yaw * relative_forward + cos_yaw * relative_left;
    if heading_forward <= 0.0 {
        return None;
    }
    let (sin_pitch, cos_pitch) = calibration.pitch_rad.sin_cos();
    let (sin_roll, cos_roll) = calibration.roll_rad.sin_cos();

    let x_camera_level = -heading_left;
    let y_camera_pitched = calibration.height_m * cos_pitch - heading_forward * sin_pitch;
    let z_camera = calibration.height_m * sin_pitch + heading_forward * cos_pitch;
    if !z_camera.is_finite() || z_camera <= 1e-4 {
        return None;
    }

    let x_camera = cos_roll * x_camera_level + sin_roll * y_camera_pitched;
    let y_camera = -sin_roll * x_camera_level + cos_roll * y_camera_pitched;
    let u = calibration.fx * x_camera / z_camera + calibration.cx;
    let v = calibration.fy * y_camera / z_camera + calibration.cy;
    (u.is_finite() && v.is_finite()).then_some((u, v))
}

fn encode_bev(bev_classes: &Mat, config: BevConfig) -> Result<BevProjection, String> {
    let mut occupancy = Mat::new_rows_cols_with_default(
        config.height,
        config.width,
        core::CV_8UC1,
        Scalar::all(OCCUPIED as f64),
    )
    .map_err(|e| e.to_string())?;
    let plane_len = (config.width * config.height) as usize;
    let mut semantic_chw = vec![0u8; BEV_SEMANTIC_CHANNELS * plane_len];

    for row in 0..config.height {
        let src = bev_classes.ptr(row).map_err(|e| e.to_string())?;
        let dst = occupancy.ptr_mut(row).map_err(|e| e.to_string())?;
        let src = unsafe { std::slice::from_raw_parts(src, config.width as usize) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, config.width as usize) };
        for col in 0..config.width as usize {
            let class_id = src[col];
            // Cityscapes train-id 0 是 road。其余类别和未知区域全部保守视为占用。
            dst[col] = if class_id == 0 { FREE } else { OCCUPIED };
            let channel = semantic_channel(class_id);
            let pixel = row as usize * config.width as usize + col;
            semantic_chw[channel * plane_len + pixel] = 1;
        }
    }

    Ok(BevProjection {
        occupancy,
        semantic_chw,
    })
}

fn semantic_channel(class_id: u8) -> usize {
    match class_id {
        0 => 0,        // road
        1 => 1,        // sidewalk
        2 => 2,        // building
        3 => 3,        // wall
        4 => 4,        // fence
        5 => 5,        // pole
        6 | 7 => 6,    // traffic light/sign
        8 => 7,        // vegetation
        9 => 8,        // terrain
        11 | 12 => 9,  // person/rider
        13 => 10,      // car
        14..=16 => 11, // heavy vehicle
        17 | 18 => 12, // motorcycle/bicycle
        _ => 13,       // sky、ignore、越界及未知
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> CameraCalibration {
        CameraCalibration {
            image_width: 640,
            image_height: 480,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            forward_offset_m: 0.07,
            left_offset_m: 0.0,
            height_m: 0.20,
            yaw_rad: 0.05,
            pitch_rad: 0.20,
            roll_rad: 0.0,
        }
    }

    fn config() -> BevConfig {
        BevConfig {
            width: 192,
            height: 192,
            meters_per_cell: 20.0 / 192.0,
            min_forward_m: 0.20,
            ego_row: 95.5,
            ego_col: 95.5,
        }
    }

    #[test]
    fn ground_projection_respects_left_right_axis() {
        let left = project_ground_point(calibration(), 2.0, 0.5).unwrap();
        let right = project_ground_point(calibration(), 2.0, -0.5).unwrap();
        assert!(left.0 < calibration().cx);
        assert!(right.0 > calibration().cx);
    }

    #[test]
    fn pixel_ground_round_trip_is_metric() {
        let expected = (2.0, 0.35);
        let pixel = project_ground_point(calibration(), expected.0, expected.1).unwrap();
        let actual = calibration().pixel_to_ground(pixel.0, pixel.1).unwrap();
        assert!((actual.0 - expected.0).abs() < 1e-4);
        assert!((actual.1 - expected.1).abs() < 1e-4);
    }

    #[test]
    fn only_road_is_free_and_semantics_are_one_hot() {
        let classes = Mat::from_slice_2d(&[[0u8, 11, 14, 255]]).unwrap();
        let result = encode_bev(
            &classes,
            BevConfig {
                width: 4,
                height: 1,
                meters_per_cell: 1.0,
                min_forward_m: 0.0,
                ego_row: 0.0,
                ego_col: 1.5,
            },
        )
        .unwrap();
        let row = unsafe { std::slice::from_raw_parts(result.occupancy.ptr(0).unwrap(), 4) };
        assert_eq!(row, &[FREE, OCCUPIED, OCCUPIED, OCCUPIED]);
        for pixel in 0..4 {
            let active = (0..BEV_SEMANTIC_CHANNELS)
                .filter(|channel| result.semantic_chw[channel * 4 + pixel] == 1)
                .count();
            assert_eq!(active, 1);
        }
    }

    #[test]
    fn project_rejects_wrong_input_shape() {
        let projector = IpmProjector::new(calibration(), config()).unwrap();
        let wrong =
            Mat::new_rows_cols_with_default(10, 10, core::CV_8UC1, Scalar::all(0.0)).unwrap();
        assert!(projector.project(&wrong).is_err());
    }

    #[test]
    fn projector_rejects_calibration_without_visible_ground() {
        let mut invalid = calibration();
        invalid.pitch_rad = -1.5;
        assert!(IpmProjector::new(invalid, config()).is_err());
    }

    #[test]
    fn deployment_crop_is_twenty_meters_square() {
        let config = config();
        assert!((config.physical_width_m() - 20.0).abs() < 1e-5);
        assert!((config.physical_forward_m() - 20.0).abs() < 1e-5);
        assert_eq!((config.ego_row, config.ego_col), (95.5, 95.5));
    }
}
