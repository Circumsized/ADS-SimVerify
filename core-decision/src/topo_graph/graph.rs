use super::node::TopologicalNode;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{Read, Write};

/// 拓扑有向边:规定了地标站牌之间的通行物理开销
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Edge {
    pub target_id: u32,    // 目标地标节点 ID
    pub weight: f32,       // 两点之间的物理距离 (作为 A* 算法的 G 权重值)
    pub relative_yaw: f32, // 驶向目标节点所期望的相对偏航角 (单位: 度)
}

/// 拓扑地图:管理所有的度量-拓扑混合地标
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TopologicalGraph {
    pub nodes: HashMap<u32, TopologicalNode>,
    pub adjacency_list: HashMap<u32, Vec<Edge>>, // 邻接表:节点ID -> 相连的有向边
}

/// 用于 A* 优先队列的辅助状态结构体
#[derive(Copy, Clone, PartialEq)]
struct State {
    estimated_total_cost: f32,
    path_cost: f32,
    position: u32,
}

impl Eq for State {}

// 实现 Ord,使 BinaryHeap 表现为最小堆(每次弹回代价最小的路径点)
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .estimated_total_cost
            .partial_cmp(&self.estimated_total_cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TopologicalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adjacency_list: HashMap::new(),
        }
    }

    /// 往地图中注册全新的空间地标
    pub fn add_node(&mut self, node: TopologicalNode) {
        self.nodes.insert(node.id, node);
    }

    /// 往地图中铺设一条有向道路
    pub fn add_edge(&mut self, from: u32, to: u32, weight: f32, relative_yaw: f32) {
        self.adjacency_list.entry(from).or_default().push(Edge {
            target_id: to,
            weight,
            relative_yaw,
        });
    }

    /// A* 全局路径寻路引擎:结合启发式几何距离,快速规划最优站牌路径
    pub fn find_path_astar(&self, start: u32, goal: u32) -> Option<Vec<u32>> {
        let mut dist: HashMap<u32, f32> = HashMap::new();
        let mut parent: HashMap<u32, u32> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start, 0.0);
        heap.push(State {
            estimated_total_cost: 0.0,
            path_cost: 0.0,
            position: start,
        });

        let goal_node = self.nodes.get(&goal)?;

        while let Some(State {
            path_cost: current_cost,
            position,
            ..
        }) = heap.pop()
        {
            if position == goal {
                let mut path = Vec::new();
                let mut curr = goal;
                while curr != start {
                    path.push(curr);
                    curr = *parent.get(&curr)?;
                }
                path.push(start);
                path.reverse();
                return Some(path);
            }

            if let Some(&d) = dist.get(&position) {
                if current_cost > d {
                    continue;
                }
            }

            if let Some(edges) = self.adjacency_list.get(&position) {
                for edge in edges {
                    let next = edge.target_id;
                    let Some(next_node) = self.nodes.get(&next) else {
                        continue;
                    };
                    if !edge.weight.is_finite() || edge.weight < 0.0 {
                        continue;
                    }
                    let new_dist =
                        dist.get(&position).copied().unwrap_or(f32::INFINITY) + edge.weight;

                    if new_dist < dist.get(&next).copied().unwrap_or(f32::INFINITY) {
                        dist.insert(next, new_dist);
                        parent.insert(next, position);
                        // A* 核心:计算度量空间下的几何欧氏距离作为启发式估算 (H 值)
                        let h = ((next_node.pose.x - goal_node.pose.x).powi(2)
                            + (next_node.pose.y - goal_node.pose.y).powi(2))
                        .sqrt();
                        heap.push(State {
                            estimated_total_cost: new_dist + h,
                            path_cost: new_dist,
                            position: next,
                        });
                    }
                }
            }
        }
        None
    }

    /// 将带有视觉描述子和北斗坐标的混合拓扑地图持久化写入硬盘
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(serialized.as_bytes())
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// 从硬盘恢复混合拓扑地图记忆
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| e.to_string())?;
        let graph: Self = serde_json::from_str(&contents).map_err(|e| e.to_string())?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo_graph::node::{Pose, TopologicalNode};

    fn node(id: u32, x: f32, y: f32) -> TopologicalNode {
        TopologicalNode {
            id,
            pose: Pose { x, y, yaw: 0.0 },
            ..Default::default()
        }
    }

    #[test]
    fn astar_skips_dangling_edges_and_keeps_valid_route() {
        let mut graph = TopologicalGraph::new();
        graph.add_node(node(0, 0.0, 0.0));
        graph.add_node(node(1, 1.0, 0.0));
        graph.add_node(node(2, 2.0, 0.0));

        graph.add_edge(0, 99, 0.1, 0.0);
        graph.add_edge(0, 1, 1.0, 0.0);
        graph.add_edge(1, 2, 1.0, 0.0);

        assert_eq!(graph.find_path_astar(0, 2), Some(vec![0, 1, 2]));
    }

    #[test]
    fn astar_ignores_invalid_edge_weights() {
        let mut graph = TopologicalGraph::new();
        graph.add_node(node(0, 0.0, 0.0));
        graph.add_node(node(1, 1.0, 0.0));
        graph.add_node(node(2, 2.0, 0.0));

        graph.add_edge(0, 1, f32::NAN, 0.0);
        graph.add_edge(0, 2, 3.0, 0.0);
        graph.add_edge(2, 1, 1.0, 0.0);

        assert_eq!(graph.find_path_astar(0, 1), Some(vec![0, 2, 1]));
    }
}
