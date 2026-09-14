/*
=================================================================
DORA reflex timing diagnostic node (async)
Millisecond timing audit | Transport jitter statistics
=================================================================
*/

use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::eyre;
use std::time::Instant;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    println!("[Rust diag] Reflex timing audit node started...");

    // 1. 接入数据流网络
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let mut loop_count: u64 = 0;
    let mut last_frame_recv_time = Instant::now();
    let mut delta_accum_ms = 0.0;
    let mut max_delta_ms = 0.0;
    let mut min_delta_ms = f64::MAX;

    // 2. 接收事件循环 (使用异步 .await 实现非阻塞高效装载)
    while let Some(event) = events.recv_async().await {
        match event {
            Event::Input { id, data, .. } => {
                if id.as_str() == "obstacle_force" {
                    loop_count += 1;
                    let now = Instant::now();

                    // 计算与上一帧到达的真实物理时间差 (Jitter 诊断)
                    let interval_ms =
                        now.duration_since(last_frame_recv_time).as_secs_f64() * 1000.0;
                    last_frame_recv_time = now;

                    if loop_count > 5 {
                        // 略过前 5 帧的温启动抖动
                        delta_accum_ms += interval_ms;
                        if interval_ms > max_delta_ms {
                            max_delta_ms = interval_ms;
                        }
                        if interval_ms < min_delta_ms {
                            min_delta_ms = interval_ms;
                        }
                    }

                    // 解析虚拟势场力
                    let raw_data_vec: Vec<u8> = dora_node_api::into_vec(&data)
                        .map_err(|e| eyre!("Failed to parse DORA data: {}", e))?;

                    if raw_data_vec.len() >= 8 {
                        let _f_x = f32::from_le_bytes(raw_data_vec[0..4].try_into().unwrap());
                        let _f_y = f32::from_le_bytes(raw_data_vec[4..8].try_into().unwrap());
                    }

                    // 每 100 帧输出一次时序统计报告
                    if loop_count % 100 == 0 && loop_count > 5 {
                        let avg_interval = delta_accum_ms / 95.0;
                        println!(
                            "[Rust timing probe] frames: {:<6} | avg interval: {:.2} ms | Jitter: [{:.2} ms - {:.2} ms]",
                            loop_count, avg_interval, min_delta_ms, max_delta_ms
                        );
                        // 重置统计滑动窗口
                        delta_accum_ms = 0.0;
                        max_delta_ms = 0.0;
                        min_delta_ms = f64::MAX;
                    }

                    // 二进制回传 Echo 指令:驱动小车向前缓行 (v=0.1, w=0.0)
                    let motion_cmd_raw_array = [0.1f32, 0.0f32];
                    let raw_memory_slice: &[u8] = unsafe {
                        std::slice::from_raw_parts(motion_cmd_raw_array.as_ptr() as *const u8, 8)
                    };

                    if let Err(e) = node.send_output_bytes(
                        "control_cmd".to_string().into(),
                        MetadataParameters::default(),
                        8,
                        raw_memory_slice,
                    ) {
                        eprintln!("Echo feedback command failed: {}", e);
                    }
                }
            }
            Event::Stop(_) => {
                println!("[Rust diag] Received DORA stop signal, exiting.");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
