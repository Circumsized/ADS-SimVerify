#!/usr/bin/env python3
import unittest

from contracts.phase5.phase5e_real_calibration_audit import validate_calibration


class Phase5ECalibrationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "image_size": [640, 480],
            "device_identity_required": ["vendor_id", "product_id", "serial"],
            "intrinsics": {
                "minimum_checkerboard_views": 20,
                "minimum_occupied_image_bins_3x3": 6,
                "rms_reprojection_error_px_max": 0.5,
                "per_view_p95_error_px_max": 0.8,
            },
            "extrinsics": {
                "rotation_orthogonality_error_max": 0.001,
                "translation_uncertainty_m_max": 0.002,
                "rotation_uncertainty_deg_max": 0.5,
            },
            "metric_depth": {
                "plane_scale_relative_error_max": 0.02,
                "paired_frame_ratio_min": 0.99,
            },
        }
        self.calibration = {
            "schema_version": "phase5e-real-camera-calibration-v1",
            "device_identity": {"vendor_id": "1234", "product_id": "5678", "serial": "abc"},
            "image_size": [640, 480],
            "intrinsics": {
                "camera_matrix": [[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "quality": {"checkerboard_views": 24, "occupied_image_bins_3x3": 8, "rms_reprojection_error_px": 0.3, "per_view_p95_error_px": 0.6},
            },
            "extrinsics": {
                "T_body_camera": [[1.0, 0.0, 0.0, 0.07], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.13], [0.0, 0.0, 0.0, 1.0]],
                "target_based_measurement": True,
                "translation_uncertainty_m": 0.001,
                "rotation_uncertainty_deg": 0.2,
            },
            "metric_depth": {"registered_to_rgb": True, "units": "meters", "plane_scale_relative_error": 0.01, "paired_frame_ratio": 1.0},
            "evidence": [],
        }

    def test_complete_calibration_passes(self):
        self.assertEqual(validate_calibration(self.calibration, self.contract), [])

    def test_non_orthonormal_extrinsic_is_rejected(self):
        self.calibration["extrinsics"]["T_body_camera"][0][0] = 1.2
        errors = validate_calibration(self.calibration, self.contract)
        self.assertTrue(any("orthonormal" in error or "determinant" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
