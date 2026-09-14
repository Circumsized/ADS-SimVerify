# core-perception 模块索引

> 当前主链使用 `XfeatExtractor`(640x640, 2 Hz) 与 `IpmProjector`(320x240 → 192x192 BEV);
> `PidnetEngine` 与 `SubpixelRefiner` 当前处于 **inactive**,不在主链中
> (见 `README.md` 与 `HANDOFF.md` 第 8 节)。

## 模块职责

| 模块 | 公开类型 | 作用范围 |
|---|---|---|
| `lib.rs` | `BiomimeticFeatureExtractor`, `BiomimeticSegmenter` | 上层抽象 trait;只暴露给同 workspace 调用方 |
| `perception::ipm_projector` | `CameraCalibration`, `BevConfig`, `IpmProjector`, `BevProjection` | 针孔相机几何 + BEV 生成(20 Hz 主链) |
| `perception::matcher` | `BiomimeticMatcher`, `HomographyEstimate`, `PlanarMetricTransform`, `SubpixelRefiner` | XFeat 匹配、单应/基础矩阵、地平面度量变换 |
| `perception::xfeat_engine` | `XfeatExtractor`, `SparseFeature` | XFeat ONNX 推理,640x640,2 Hz(慢通道) |
| `perception::pidnet_engine` | `PidnetEngine` | Cityscapes PIDNet(仓库保留为负对照,**不接主链**) |
| `lib.rs::self_heal_load_onnx_dylib` | 函数 | 自动探测 `libonnxruntime.so` 路径,注入 `ORT_DYLIB_PATH` |

## 主链调用关系

```text
Dora jpeg_image
   -> perception_node (bin)
      -> IpmProjector.project(class_map)              // 主链 20 Hz, 192x192 BEV
      -> XfeatExtractor.extract_features(image)      // 慢通道 2 Hz, 200 关键点上限
   -> Dora bev_grid / xfeat_features
```

## 与 `contracts/v1/data_contracts.json` 的对应

* `image.jpeg.v1` → `IpmProjector` / `XfeatExtractor` 输入
* `bev.occupancy.perception.v1` → `IpmProjector.project()` 输出
* `bev.semantic.cityscapes14.v1` → `IpmProjector::encode_bev()` 输出
* `features.xfeat64.v1` → `XfeatExtractor.extract_features()` 输出

## 测试与禁用矩阵

* `pidnet_engine` 模块默认不参与测试,因为 `model/pidnet_s.onnx` 不在仓库中;
* `XfeatExtractor::new` 的真实模型测试通过 `MODEL_TEST_LOCK` 串行化,
  避免同一 GPU 上的并发 ORT session 抢占;
* `cargo test -p core-perception` 在没有 `model/xfeat_640x640.onnx` 的
  容器/CI 环境里只跑几何/算子层单元测试,XFeat 真实推理测试会被跳过。