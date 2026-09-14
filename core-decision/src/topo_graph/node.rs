use serde::{Deserialize, Serialize};

/// 物理姿态量:兼容局部相对里程计和北斗全局坐标系
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub struct Pose {
    pub x: f32,   // 绝对 UTM Northing 或局部相对坐标 X
    pub y: f32,   // 绝对 UTM Easting 或局部相对坐标 Y
    pub yaw: f32, // 车头偏航角 (弧度)
}

/// 拓扑空间地标节点 (Topometric Node)
/// 融合度量坐标与视觉稀疏特征指纹
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TopologicalNode {
    pub id: u32,      // 唯一的节点站牌 ID
    pub name: String, // 节点语义名称 (如 "5栋楼下", "地下室B1拐角")
    pub pose: Pose,   // 拍照时的绝对坐标(北斗经纬度投射值或里程计)
    /// XFeat 描述子集合:扁平化的描述子特征向量 [N_features * 64 维]
    pub descriptors: Vec<f32>,
    /// 每个特征点的 (x, y) 像素物理坐标:扁平化的坐标矩阵 [N_features * 2]
    pub keypoints: Vec<f32>,
}
