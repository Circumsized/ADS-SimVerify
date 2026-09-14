# Phase 7 硬件规格与部署画像

## 1. 结论

正式推荐 **NVIDIA Jetson Orin NX 16GB Super**，配合主动散热、可靠载板、NVMe、
独立安全 MCU 和具备公制深度输出的 RGB-D 相机。Orin Nano Super 8GB 只能作为
最低原型；AGX Orin 32GB 适合未来同时增加多相机、更大模型或高分辨率录制的情况。

本结论允许采购和板端适配，不允许直接控制真车。目标板必须再次运行同一套 Phase 7
profile，并通过 20 Hz 主链 p95 小于 50 ms、NMPC p95 小于 10 ms、CUDA provider
真实生效等门禁。

## 2. 实际测量了什么

测量环境为 i5-13600KF、32GB RAM、RTX 3060 12GB；Isaac Sim 未启动。输入是冻结的
100 帧 640x480 JPEG 与 float32 公制深度，避免把渲染器显存算进车载需求。

| 路径 | 样本 | p50 | p95 | 最大值 |
|---|---:|---:|---:|---:|
| JPEG 解码 | 100 | 0.65 ms | 1.02 ms | 1.28 ms |
| warehouse_nav14 预处理 | 100 | 0.69 ms | 0.88 ms | 0.99 ms |
| warehouse_nav14 CUDA 推理 | 100 | 5.05 ms | 5.84 ms | 8.87 ms |
| depth-lift BEV | 100 | 7.11 ms | 8.26 ms | 8.61 ms |
| acados NMPC | 100 | 0.63 ms | 1.20 ms | 1.24 ms |
| 完整控制主链 | 100 | 15.14 ms | **17.07 ms** | 19.24 ms |
| XFeat 640x640 慢路径 | 20 | 3.61 ms | **3.85 ms** | 4.07 ms |

CPU 回退对照中，语义 ONNX p95 为 15.11 ms，XFeat ONNX p95 为 31.27 ms。这只能证明
故障降级具备计算可行性，不代表允许长期脱离 CUDA 运行。

进程 RSS 从 236.97 MiB 上升到峰值 1,448.95 MiB，增量为 1,211.97 MiB。桌面 GPU
显存从采样基线增加到峰值增量 253 MiB。两个 CUDA session 和两个 CPU 回退 session
同时驻留，因此这个 RSS 结果比只加载正式 CUDA session 更保守。

## 3. 为什么不是直接买 8GB

可以把统一内存想成一张工作台。模型文件本身只有约 16.1 MiB，但运行时还要在工作台
上同时展开中间特征、TensorRT workspace、Dora 队列、RGB/深度缓冲、操作系统和日志。
桌面实测的 1.18 GiB 增量只是当前短 profile 的进程峰值，不包含 Jetson 图形桌面、相机
驱动和完整板端服务。8GB 能做实验，但一旦增加录制、多相机或构建 TensorRT engine，
余量会迅速缩小。

16GB 的 Orin NX 像是尺寸合适且仍留有空位的工作台：157 INT8 TOPS、102 GB/s 内存
带宽、最高 40W，满足冻结的 16GB、100 TOPS、100 GB/s 和 40W 上限。32GB AGX
Orin 不是当前必需，它主要购买的是未来扩展余量和更高带宽。

## 4. 数据流与带宽

按当前未压缩内存布局声明的带宽是：RGB 18.432 MB/s、公制深度 24.576 MB/s、
14 通道 BEV 41.288 MB/s、2 Hz XFeat 张量 3.277 MB/s，合计 **87.572 MB/s**。
它不是 USB 线速，也不是磁盘写入速度，而是节点之间若复制完整张量时的理论负载。
因此 Dora 应继续使用有界队列与零拷贝路径，录制节点不能阻塞 20 Hz 控制链。

## 5. 推荐硬件画像

| 层级 | 计算模块 | 定位 |
|---|---|---|
| 最低原型 | Jetson Orin Nano Super 8GB | 只做台架验证，不作为正式车载推荐 |
| 正式推荐 | **Jetson Orin NX 16GB Super** | 当前单相机、20 Hz 感知 NMPC 与 2 Hz XFeat |
| 高余量 | Jetson AGX Orin 32GB | 多相机、更大模型、高码率长期录制 |

建议配套：

- 可靠的 Orin NX 载板和匹配的主动散热，按 40W 模块功耗再留外设与瞬态余量。
- 512GB 起步、建议 1TB 的工业或高耐久 NVMe，用于模型、日志和回放数据。
- RealSense D455 作为首个 RGB-D 候选：USB-C 3.1 Gen 1、深度/RGB 全局快门；购买后
  仍必须冻结真机内参、畸变、相机到车体外参和深度尺度。
- 独立 MCU 负责电机闭环、物理急停、心跳超时归零；Jetson 不应成为唯一安全边界。
- 稳压电源、保险/过流保护和可测量的整机功耗；40W 是模块档位，不是整车电源预算。

开发套件适合原型，量产或长期车载应使用正式模块、载板和热设计。Jetson Orin 模块官方
生命周期到 2032 年 1 月，适合当前项目继续工程化。

## 6. 不能从本次测量推出什么

RTX 3060 的 17.07 ms 不能按 TOPS 比例换算成 Orin 延迟；GPU 架构、内存共享、
TensorRT engine 和功耗模式都不同。XFeat 测量包含 letterbox、归一化和 ONNX 推理，
不包含 Rust 稀疏关键点后处理。45.85W 的峰值是桌面 GPU 采样，不是 Jetson 或整机功耗。

通俗地说，本次 profile 像是在实验室称出了行李的重量，足以选择合适尺寸的箱子；但箱子
买回来后仍要真正装箱、过秤和试走，不能拿实验室秤的数字当作机场已经放行。

## 7. 目标板到货后的唯一门禁

1. 安装当前 Orin 支持的 JetPack 6 分支、TensorRT、ONNX Runtime 和 acados，锁定版本。
2. 将 warehouse_nav14 与 XFeat 转成目标板 TensorRT engine，保留 ONNX 回退。
3. 接入 D455，完成内参、畸变、外参、时间戳和 depth scale 标定。
4. 重跑 `run_phase7_profile.py` 等价板端采集，验证 provider、p95、RSS、统一显存和功耗。
5. 重跑 Phase 5-I/5-K watchdog、急停、故障恢复与小时耐久门禁。
6. 以上全部通过后，才允许进入低速、架空轮、系绳和封闭场地的分级真机验证。

## 8. 官方规格来源

- Jetson Orin 系列规格：https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- Orin Nano Super：https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- Jetson 生命周期：https://developer.nvidia.com/embedded/lifecycle
- Jetson 开发套件与模块边界：https://developer.nvidia.com/embedded/faq
- RealSense D455：https://www.realsenseai.com/products/real-sense-depth-camera-d455f/
- TensorRT 支持矩阵：https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html

原始证据：`artifacts/phase7_profile/final_20260716_174458/summary.json`。
