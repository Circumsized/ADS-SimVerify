use core_control::monotonic_ms;
use core_safety::{ControlProposal, RuntimeHealth, SafetyState, SafetySupervisor, SupervisorInput};
use dora_node_api::arrow::array::{Float32Array, UInt8Array};
use dora_node_api::{DoraNode, Event, Metadata, MetadataParameters, Parameter};

fn param_bool(metadata: &Metadata, key: &str) -> bool {
    match metadata.parameters.get(key) {
        Some(Parameter::Bool(value)) => *value,
        _ => false,
    }
}

fn param_i64(metadata: &Metadata, key: &str) -> i64 {
    match metadata.parameters.get(key) {
        Some(Parameter::Integer(value)) => *value,
        _ => 0,
    }
}

fn param_f64(metadata: &Metadata, key: &str) -> f64 {
    match metadata.parameters.get(key) {
        Some(Parameter::Float(value)) => *value,
        _ => f64::NAN,
    }
}

fn state_name(state: SafetyState) -> &'static str {
    match state {
        SafetyState::Boot => "boot",
        SafetyState::Warmup => "warmup",
        SafetyState::Ready => "ready",
        SafetyState::Active => "active",
        SafetyState::EmergencyStop => "emergency_stop",
        SafetyState::Fault => "fault",
        SafetyState::Stopped => "stopped",
    }
}

fn state_code(state: SafetyState) -> i32 {
    match state {
        SafetyState::Boot => 0,
        SafetyState::Warmup => 1,
        SafetyState::Ready => 2,
        SafetyState::Active => 3,
        SafetyState::EmergencyStop => 4,
        SafetyState::Fault => 5,
        SafetyState::Stopped => 6,
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut supervisor = SafetySupervisor::new(5, 150.0, 0.8, 0.6).map_err(|e| eyre::eyre!(e))?;
    let mut emergency_stop = false;
    let mut reset_pending = false;

    while let Some(event) = events.recv_async().await {
        match event {
            Event::Input { id, metadata, data } if id.as_str() == "control_cmd" => {
                let (linear, angular) = data
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .filter(|array| array.len() == 2)
                    .map(|array| (array.value(0) as f64, array.value(1) as f64))
                    .unwrap_or((f64::NAN, f64::NAN));

                let frame_id = param_i64(&metadata, "source_frame_id").max(0) as u64;
                let epoch = param_i64(&metadata, "session_epoch").max(0) as u64;
                let health_timestamp_ms = param_f64(&metadata, "health_timestamp_ms");
                let proposal_timestamp_ms = param_f64(&metadata, "proposal_timestamp_ms");

                let health = RuntimeHealth {
                    frame_id,
                    timestamp_ms: health_timestamp_ms,
                    sensor_valid: param_bool(&metadata, "sensor_valid"),
                    perception_valid: param_bool(&metadata, "perception_valid"),
                    solver_valid: param_bool(&metadata, "solver_valid"),
                    articulation_ready: param_bool(&metadata, "articulation_ready"),
                };
                let command = ControlProposal {
                    frame_id,
                    timestamp_ms: proposal_timestamp_ms,
                    linear,
                    angular,
                };

                let now_ms = monotonic_ms();
                let decision = supervisor.step(SupervisorInput {
                    now_ms,
                    epoch,
                    health: Some(health),
                    command: Some(command),
                    emergency_stop,
                    reset: reset_pending,
                    shutdown: false,
                });
                reset_pending = false;

                let mut out_metadata = MetadataParameters::new();
                out_metadata.insert(
                    "source_frame_id".into(),
                    Parameter::Integer(frame_id as i64),
                );
                out_metadata.insert(
                    "safety_state".into(),
                    Parameter::String(state_name(decision.state).into()),
                );
                out_metadata.insert(
                    "safety_reason".into(),
                    Parameter::String(decision.reason.clone()),
                );
                out_metadata.insert("safety_latched".into(), Parameter::Bool(decision.latched));
                out_metadata.insert("supervisor_timestamp_ms".into(), Parameter::Float(now_ms));

                node.send_output(
                    "safe_control".to_string().into(),
                    out_metadata,
                    Float32Array::from(vec![decision.linear as f32, decision.angular as f32]),
                )?;
                node.send_output(
                    "safety_state".to_string().into(),
                    MetadataParameters::new(),
                    Float32Array::from(vec![
                        frame_id as f32,
                        state_code(decision.state) as f32,
                        decision.linear as f32,
                        decision.angular as f32,
                    ]),
                )?;
            }
            Event::Input { id, data, .. } if id.as_str() == "safety_request" => {
                if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                    if array.len() >= 2 {
                        emergency_stop = array.value(0) != 0;
                        reset_pending |= array.value(1) != 0;
                    }
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
