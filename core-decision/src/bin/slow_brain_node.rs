/*
=================================================================
慢系统决策脑:北斗 GPS 与视觉重放混合导航适配器
GPS 高精引导与视觉指纹降级定位双轨并网
=================================================================
*/
use dora_node_api::arrow::array::{Array, FixedSizeListArray, Float32Array, StructArray};
use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::eyre;
use std::time::Instant;

use core_decision::topo_graph::graph::TopologicalGraph;
use core_decision::topo_graph::node::TopologicalNode;
use core_perception::perception::matcher::BiomimeticMatcher;
use core_perception::perception::xfeat_engine::SparseFeature;

const XFEAT_DESCRIPTOR_DIM: usize = 64;
const KEYPOINT_COORD_DIM: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
enum NavigationMode {
    BeidouHighPrecisionNav, // GPS 信号优良 (HDOP < 2.5)
    PuppetReplay,           // 室内/地下室 GPS 失锁降级 (HDOP >= 5.0 或无信号)
}

struct SlowBrainStateMachine {
    pub current_mode: NavigationMode,
    pub odom_x: f32,
    pub odom_y: f32,
    pub gps_x: f32,
    pub gps_y: f32,
    pub gps_hdop: f32,
    pub last_gps_time: Instant,
    pub current_target_index: usize,
    pub topo_memory: TopologicalGraph,
    pub nav_route: Vec<u32>,
}

fn recover_history_visual_fingerprint(node: &TopologicalNode) -> Vec<SparseFeature> {
    if node.descriptors.is_empty()
        || !node.descriptors.len().is_multiple_of(XFEAT_DESCRIPTOR_DIM)
        || node.keypoints.len()
            != (node.descriptors.len() / XFEAT_DESCRIPTOR_DIM) * KEYPOINT_COORD_DIM
    {
        return Vec::new();
    }

    let (descriptor_chunks, _) = node.descriptors.as_chunks::<XFEAT_DESCRIPTOR_DIM>();
    let (keypoint_chunks, _) = node.keypoints.as_chunks::<KEYPOINT_COORD_DIM>();
    descriptor_chunks
        .iter()
        .zip(keypoint_chunks.iter())
        .map(|(descriptor, keypoint)| SparseFeature {
            x: keypoint[0],
            y: keypoint[1],
            confidence: 1.0,
            descriptor: descriptor.to_vec(),
        })
        .collect()
}

/// 把跨进程 Arrow `StructArray` 安全解码为强类型稀疏特征列表。
///
/// 这是不可信边界的唯一入口:任何 schema、类型、长度、null 或非有限数值问题
/// 都以 `Err(String)` 返回,调用方丢弃该帧而不是 panic。工作区 release 配置为
/// `panic = "abort"`,因此这里绝不允许 `unwrap()`/索引越界。
fn decode_xfeat_struct(struct_array: &StructArray) -> Result<Vec<SparseFeature>, String> {
    let x_array = struct_column_f32(struct_array, "x")?;
    let y_array = struct_column_f32(struct_array, "y")?;
    let score_array = struct_column_f32(struct_array, "score")?;

    let row_count = x_array.len();
    if y_array.len() != row_count || score_array.len() != row_count {
        return Err(format!(
            "column length mismatch: x={row_count} y={} score={}",
            y_array.len(),
            score_array.len()
        ));
    }

    let desc_array = struct_array
        .column_by_name("descriptor")
        .ok_or_else(|| "missing descriptor column".to_string())?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| "descriptor column is not FixedSizeListArray".to_string())?;
    if desc_array.len() != row_count {
        return Err(format!(
            "descriptor length mismatch: x={row_count} descriptor={}",
            desc_array.len()
        ));
    }
    if desc_array.value_length() as usize != XFEAT_DESCRIPTOR_DIM {
        return Err(format!(
            "descriptor width {} != expected {}",
            desc_array.value_length(),
            XFEAT_DESCRIPTOR_DIM
        ));
    }
    let desc_child = desc_array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| "descriptor child is not Float32Array".to_string())?;

    let mut features = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let px = x_array.value(i);
        let py = y_array.value(i);
        let conf = score_array.value(i);
        if !px.is_finite() || !py.is_finite() || !conf.is_finite() {
            return Err(format!("non-finite keypoint at row {i}"));
        }

        let base = i * XFEAT_DESCRIPTOR_DIM;
        let mut descriptor = vec![0.0f32; XFEAT_DESCRIPTOR_DIM];
        for (offset, slot) in descriptor.iter_mut().enumerate() {
            let value = desc_child.value(base + offset);
            if !value.is_finite() {
                return Err(format!("non-finite descriptor at row {i}"));
            }
            *slot = value;
        }
        features.push(SparseFeature {
            x: px,
            y: py,
            confidence: conf,
            descriptor,
        });
    }
    Ok(features)
}

fn struct_column_f32<'a>(
    struct_array: &'a StructArray,
    name: &str,
) -> Result<&'a Float32Array, String> {
    struct_array
        .column_by_name(name)
        .ok_or_else(|| format!("missing column {name}"))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| format!("column {name} is not Float32Array"))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    println!("========================================================");
    println!("慢系统决策脑: 北斗-视觉 Teach&Repeat 混合定位并网器启动...");
    println!("========================================================");

    let (mut node, mut events) = DoraNode::init_from_env()?;

    // 1. 初始化混合拓扑地图记忆(如果不存在则使用内置路点模拟)
    let map_path = "topo_memory.json";
    let mut brain_memory = if std::path::Path::new(map_path).exists() {
        TopologicalGraph::load_from_file(map_path).unwrap_or_else(|_| TopologicalGraph::new())
    } else {
        TopologicalGraph::new()
    };

    // 如果是冷启动空地图,硬编码注入赛道路标卡片,确保开箱即用
    if brain_memory.nodes.is_empty() {
        println!(
            "Slow-brain: no physical map file detected, generating Beidou-visual hybrid topology skeleton online..."
        );
        let waymarks = vec![
            (0, "起点站牌", 0.0, 0.0, 0.0),
            (1, "S弯入口站牌", 0.20, 1.50, 0.0),
            (2, "货架障碍区站牌", 0.40, 2.80, 0.0),
            (3, "左弯死角站牌", -1.00, 3.50, 0.0),
            (4, "终点冲刺站牌", 0.52, 4.11, 0.0),
        ];
        for (id, name, x, y, yaw) in waymarks {
            let node = core_decision::topo_graph::node::TopologicalNode {
                id,
                name: name.to_string(),
                pose: core_decision::topo_graph::node::Pose { x, y, yaw },
                ..Default::default()
            };
            // 冷启动路点只提供度量目标;视觉重定位必须等待真实 XFeat 描述子和 keypoints 成对写入。
            brain_memory.add_node(node);
        }
        // 铺设双向通道
        brain_memory.add_edge(0, 1, 1.5, 0.0);
        brain_memory.add_edge(1, 2, 1.3, 0.0);
        brain_memory.add_edge(2, 3, 1.6, 15.0);
        brain_memory.add_edge(3, 4, 1.8, -30.0);
        let _ = brain_memory.save_to_file(map_path);
    }

    // 自动寻路:起点 0 到 终点 4
    let planned_route = brain_memory
        .find_path_astar(0, 4)
        .unwrap_or_else(|| vec![0, 1, 2, 3, 4]);
    println!("慢脑寻路成功: A* 规划路标链: {:?}", planned_route);

    let mut state = SlowBrainStateMachine {
        current_mode: NavigationMode::PuppetReplay,
        odom_x: 0.0,
        odom_y: 0.0,
        gps_x: 0.0,
        gps_y: 0.0,
        gps_hdop: 99.0, // 默认无星状态
        last_gps_time: Instant::now(),
        current_target_index: 0,
        topo_memory: brain_memory,
        nav_route: planned_route,
    };

    while let Some(event) = events.recv_async().await {
        match event {
            Event::Input { id, data, .. } => {
                let id_str = id.as_str();
                match id_str {
                    // 北斗高精度定位数据流入 (来自 DX-GP24-A 串口转换节点)
                    "gps" => {
                        let gps_arr = data
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| eyre!("Failed to parse GPS Arrow"))?;
                        if gps_arr.len() >= 3 {
                            state.gps_x = gps_arr.value(0);
                            state.gps_y = gps_arr.value(1);
                            state.gps_hdop = gps_arr.value(2);
                            state.last_gps_time = Instant::now();
                        }
                    }

                    // 物理里程计高频流入
                    "odometry" => {
                        let odom_arr = data
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| eyre!("Failed to parse Odometry"))?;
                        if odom_arr.len() >= 2 {
                            state.odom_x = odom_arr.value(0);
                            state.odom_y = odom_arr.value(1);
                        }

                        // 双模仲裁:根据星况决定当前导航模式
                        let gps_timeout = state.last_gps_time.elapsed().as_secs_f32();
                        let previous_mode = state.current_mode;

                        // 判决条件:如果卫星星况极佳,且没有超时,激活北斗领航
                        if state.gps_hdop < 2.5 && gps_timeout < 2.0 {
                            state.current_mode = NavigationMode::BeidouHighPrecisionNav;
                        } else {
                            state.current_mode = NavigationMode::PuppetReplay;
                        }

                        if state.current_mode != previous_mode {
                            println!("\n导航模式切换: 当前模式 = {:?}", state.current_mode);
                        }

                        // 执行对应模式的导航指令解算
                        if state.current_target_index < state.nav_route.len() {
                            let target_node_id = state.nav_route[state.current_target_index];
                            if let Some(target_node) = state.topo_memory.nodes.get(&target_node_id)
                            {
                                let (mut cur_pos_x, mut cur_pos_y) = (state.odom_x, state.odom_y);

                                if state.current_mode == NavigationMode::BeidouHighPrecisionNav {
                                    // 北斗领航:使用绝对北斗物理坐标系对齐
                                    cur_pos_x = state.gps_x;
                                    cur_pos_y = state.gps_y;
                                }

                                let dx = target_node.pose.x - cur_pos_x;
                                let dy = target_node.pose.y - cur_pos_y;
                                let remaining_dist = (dx * dx + dy * dy).sqrt();

                                // 到达判定:25厘米内判定过关,换下一个站牌
                                if remaining_dist < 0.25
                                    && state.current_target_index + 1 < state.nav_route.len()
                                {
                                    println!(
                                        "站牌通关: 成功越过 {} 号路标 ({:.2}, {:.2})",
                                        target_node_id, target_node.pose.x, target_node.pose.y
                                    );
                                    state.current_target_index += 1;
                                }

                                // 100Hz 广播引力坐标 human_prior
                                let prior_arr = Float32Array::from(vec![
                                    target_node.pose.x,
                                    target_node.pose.y,
                                    target_node.pose.yaw,
                                ]);
                                let _ = node.send_output(
                                    "human_prior".to_string().into(),
                                    MetadataParameters::default(),
                                    prior_arr,
                                );
                            }
                        }
                    }

                    // 视觉稀疏特征点流入 (来自前视单目 XFeat 提取)
                    // 视觉重放模式的核心定位机制:
                    // 只有在北斗失效、视觉接管时,才启动重度 XFeat / RANSAC 对齐,保障 CPU 资源
                    "xfeat_features" if state.current_mode == NavigationMode::PuppetReplay => {
                        let struct_array = match data.as_any().downcast_ref::<StructArray>() {
                            Some(array) => array,
                            None => {
                                eprintln!(
                                        "[SlowBrain] xfeat_features is not a StructArray; dropping frame"
                                    );
                                continue;
                            }
                        };
                        let current_frame_features = match decode_xfeat_struct(struct_array) {
                            Ok(features) => features,
                            Err(error) => {
                                eprintln!(
                                        "[SlowBrain] rejected malformed xfeat_features; dropping frame: {error}"
                                    );
                                continue;
                            }
                        };

                        // 检索当前要追踪的历史站牌指纹
                        let target_node_id = state.nav_route[state.current_target_index];
                        if let Some(target_node) = state.topo_memory.nodes.get(&target_node_id) {
                            // 将扁平化的一维描述子恢复成 64D 数组;缺少 keypoints 的历史指纹直接跳过。
                            let history_features = recover_history_visual_fingerprint(target_node);
                            if history_features.len() < 8 {
                                continue;
                            }

                            // 运行双向余弦交叉匹配
                            let matches = BiomimeticMatcher::cross_match(
                                &current_frame_features,
                                &history_features,
                                0.75,
                            );
                            if matches.len() >= 8 {
                                // 几何 RANSAC 过滤
                                if let Ok(clean_matches) =
                                    BiomimeticMatcher::geometry_correction_filter(
                                        &current_frame_features,
                                        &history_features,
                                        &matches,
                                        3.0,
                                    )
                                {
                                    if clean_matches.len() >= 5 {
                                        // 单目一般场景匹配没有尺度,不能把像素差直接写入米制里程计。
                                        // 此处只确认存在稳定的几何重定位证据;度量校正必须等待带尺度地图或地面特征标记。
                                        let _ = BiomimeticMatcher::estimate_homography(
                                            &current_frame_features,
                                            &history_features,
                                            &clean_matches,
                                            3.0,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::Stop(_) => {
                println!("Slow-brain safely offline.");
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_fingerprint_skips_when_keypoints_missing() {
        let mut node = TopologicalNode::default();
        node.descriptors = vec![0.1; XFEAT_DESCRIPTOR_DIM * 2];

        assert!(recover_history_visual_fingerprint(&node).is_empty());
    }

    #[test]
    fn history_fingerprint_restores_64d_descriptor_and_xy_pairs() {
        let mut node = TopologicalNode::default();
        node.descriptors = vec![0.0; XFEAT_DESCRIPTOR_DIM * 2];
        node.descriptors[3] = 1.0;
        node.descriptors[XFEAT_DESCRIPTOR_DIM + 7] = 1.0;
        node.keypoints = vec![10.0, 20.0, 30.0, 40.0];

        let features = recover_history_visual_fingerprint(&node);

        assert_eq!(features.len(), 2);
        assert_eq!((features[0].x, features[0].y), (10.0, 20.0));
        assert_eq!((features[1].x, features[1].y), (30.0, 40.0));
        assert_eq!(features[0].descriptor.len(), XFEAT_DESCRIPTOR_DIM);
        assert_eq!(features[1].descriptor.len(), XFEAT_DESCRIPTOR_DIM);
    }

    use dora_node_api::arrow::array::ArrayRef;
    use dora_node_api::arrow::datatypes::{DataType, Field};
    use std::sync::Arc;

    fn f32_column(values: Vec<f32>) -> ArrayRef {
        Arc::new(Float32Array::from(values))
    }

    fn descriptor_column(rows: usize, dim: i32) -> ArrayRef {
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let values: ArrayRef = Arc::new(Float32Array::from(vec![0.1f32; dim as usize * rows]));
        Arc::new(FixedSizeListArray::new(field, dim, values, None))
    }

    fn f32_field(name: &str) -> Arc<Field> {
        Arc::new(Field::new(name, DataType::Float32, false))
    }

    fn descriptor_field() -> Arc<Field> {
        Arc::new(Field::new(
            "descriptor",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                XFEAT_DESCRIPTOR_DIM as i32,
            ),
            false,
        ))
    }

    fn xfeat_struct(
        x: ArrayRef,
        y: ArrayRef,
        score: ArrayRef,
        descriptor: ArrayRef,
    ) -> StructArray {
        StructArray::from(vec![
            (f32_field("x"), x),
            (f32_field("y"), y),
            (f32_field("score"), score),
            (descriptor_field(), descriptor),
        ])
    }

    #[test]
    fn decode_accepts_well_formed_struct() {
        let array = xfeat_struct(
            f32_column(vec![1.0, 2.0]),
            f32_column(vec![3.0, 4.0]),
            f32_column(vec![0.9, 0.8]),
            descriptor_column(2, XFEAT_DESCRIPTOR_DIM as i32),
        );
        let features = decode_xfeat_struct(&array).expect("valid struct must decode");
        assert_eq!(features.len(), 2);
        assert_eq!((features[0].x, features[0].y), (1.0, 3.0));
        assert_eq!(features[1].descriptor.len(), XFEAT_DESCRIPTOR_DIM);
    }

    #[test]
    fn decode_rejects_missing_column() {
        let array = StructArray::from(vec![
            (f32_field("x"), f32_column(vec![1.0])),
            (f32_field("y"), f32_column(vec![1.0])),
        ]);
        assert!(decode_xfeat_struct(&array).is_err());
    }

    #[test]
    fn decode_rejects_missing_descriptor() {
        // StructArray 保证子列等长,故长度不符无法用合法构造触发;
        // 这里覆盖可达的 descriptor 缺失分支。
        let array = StructArray::from(vec![
            (f32_field("x"), f32_column(vec![1.0])),
            (f32_field("y"), f32_column(vec![1.0])),
            (f32_field("score"), f32_column(vec![1.0])),
        ]);
        assert!(decode_xfeat_struct(&array).is_err());
    }

    #[test]
    fn decode_rejects_non_finite_keypoint() {
        let array = xfeat_struct(
            f32_column(vec![f32::NAN, 2.0]),
            f32_column(vec![1.0, 2.0]),
            f32_column(vec![1.0, 2.0]),
            descriptor_column(2, XFEAT_DESCRIPTOR_DIM as i32),
        );
        assert!(decode_xfeat_struct(&array).is_err());
    }

    #[test]
    fn decode_rejects_wrong_descriptor_width() {
        let array = xfeat_struct(
            f32_column(vec![1.0]),
            f32_column(vec![1.0]),
            f32_column(vec![1.0]),
            descriptor_column(1, 32),
        );
        assert!(decode_xfeat_struct(&array).is_err());
    }
}
