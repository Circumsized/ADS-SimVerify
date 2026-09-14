use core_safety::{ControlProposal, RuntimeHealth, SafetyState, SafetySupervisor, SupervisorInput};

fn healthy(frame_id: u64, timestamp_ms: f64) -> RuntimeHealth {
    RuntimeHealth {
        frame_id,
        timestamp_ms,
        sensor_valid: true,
        perception_valid: true,
        solver_valid: true,
        articulation_ready: true,
    }
}

fn proposal(frame_id: u64, timestamp_ms: f64) -> ControlProposal {
    ControlProposal {
        frame_id,
        timestamp_ms,
        linear: 0.4,
        angular: 0.2,
    }
}

fn input(frame_id: u64, timestamp_ms: f64) -> SupervisorInput {
    SupervisorInput {
        now_ms: timestamp_ms,
        epoch: 1,
        health: Some(healthy(frame_id, timestamp_ms)),
        command: Some(proposal(frame_id, timestamp_ms)),
        emergency_stop: false,
        reset: false,
        shutdown: false,
    }
}

#[test]
fn warmup_requires_consecutive_healthy_frames_before_allowing_motion() {
    let mut supervisor = SafetySupervisor::new(3, 150.0, 0.8, 0.6).unwrap();

    let first = supervisor.step(input(1, 10.0));
    assert_eq!(first.state, SafetyState::Warmup);
    assert_eq!((first.linear, first.angular), (0.0, 0.0));

    let second = supervisor.step(input(2, 20.0));
    assert_eq!(second.state, SafetyState::Warmup);
    assert_eq!((second.linear, second.angular), (0.0, 0.0));

    let third = supervisor.step(input(3, 30.0));
    assert_eq!(third.state, SafetyState::Ready);
    assert_eq!(third.reason, "startup_ready");
    assert_eq!((third.linear, third.angular), (0.0, 0.0));

    let fourth = supervisor.step(input(4, 40.0));
    assert_eq!(fourth.state, SafetyState::Active);
    assert_eq!((fourth.linear, fourth.angular), (0.4, 0.2));
}

#[test]
fn unhealthy_runtime_faults_and_latches_zero_until_reset() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let mut unhealthy = input(1, 10.0);
    unhealthy.health.as_mut().unwrap().perception_valid = false;

    let fault = supervisor.step(unhealthy);
    assert_eq!(fault.state, SafetyState::Fault);
    assert_eq!(fault.reason, "runtime_unhealthy");
    assert!(fault.latched);

    let still_fault = supervisor.step(input(2, 20.0));
    assert_eq!(still_fault.state, SafetyState::Fault);
    assert_eq!((still_fault.linear, still_fault.angular), (0.0, 0.0));

    let mut reset = input(1, 30.0);
    reset.reset = true;
    let after_reset = supervisor.step(reset);
    assert_eq!(after_reset.state, SafetyState::Ready);
    assert!(!after_reset.latched);
}

#[test]
fn emergency_stop_latches_until_explicit_reset() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let mut stop = input(1, 10.0);
    stop.emergency_stop = true;

    let stopped = supervisor.step(stop);
    assert_eq!(stopped.state, SafetyState::EmergencyStop);
    assert!(stopped.latched);

    let still_stopped = supervisor.step(input(2, 20.0));
    assert_eq!(still_stopped.state, SafetyState::EmergencyStop);

    let mut reset = input(1, 30.0);
    reset.reset = true;
    let after_reset = supervisor.step(reset);
    assert_eq!(after_reset.state, SafetyState::Ready);
}

#[test]
fn shutdown_is_terminal_and_always_zero() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let mut shutdown = input(1, 10.0);
    shutdown.shutdown = true;

    let decision = supervisor.step(shutdown);
    assert_eq!(decision.state, SafetyState::Stopped);
    assert_eq!(decision.reason, "shutdown");

    let after = supervisor.step(input(2, 20.0));
    assert_eq!(after.state, SafetyState::Stopped);
    assert_eq!((after.linear, after.angular), (0.0, 0.0));
}

#[test]
fn validation_reasons_match_phase5_contract() {
    let cases: Vec<(&str, SupervisorInput)> = vec![
        (
            "missing_runtime_input",
            SupervisorInput {
                health: None,
                command: None,
                ..input(1, 10.0)
            },
        ),
        (
            "frame_mismatch",
            SupervisorInput {
                command: Some(proposal(2, 10.0)),
                ..input(1, 10.0)
            },
        ),
        (
            "invalid_health_timestamp",
            SupervisorInput {
                now_ms: 9.0,
                ..input(1, 10.0)
            },
        ),
        (
            "stale_health",
            SupervisorInput {
                now_ms: 200.1,
                ..input(1, 10.0)
            },
        ),
        (
            "invalid_command_timestamp",
            SupervisorInput {
                command: Some(proposal(1, f64::NAN)),
                ..input(1, 10.0)
            },
        ),
        (
            "timestamp_mismatch",
            SupervisorInput {
                now_ms: 11.0,
                command: Some(proposal(1, 11.0)),
                ..input(1, 10.0)
            },
        ),
        (
            "invalid_command",
            SupervisorInput {
                command: Some(ControlProposal {
                    linear: 0.9,
                    ..proposal(1, 10.0)
                }),
                ..input(1, 10.0)
            },
        ),
    ];

    for (reason, case) in cases {
        let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
        let decision = supervisor.step(case);
        assert_eq!(decision.reason, reason);
        assert_eq!(decision.state, SafetyState::Fault);
        assert_eq!((decision.linear, decision.angular), (0.0, 0.0));
    }
}

#[test]
fn frame_sequence_gap_faults_after_a_valid_frame() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 10.0)).state, SafetyState::Ready);

    let gap = supervisor.step(input(3, 20.0));
    assert_eq!(gap.state, SafetyState::Fault);
    assert_eq!(gap.reason, "frame_sequence_gap");
}

#[test]
fn stale_command_faults_while_health_is_fresh() {
    let mut supervisor = SafetySupervisor::new(1, 120.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 0.0)).state, SafetyState::Ready);
    assert_eq!(supervisor.step(input(2, 50.0)).state, SafetyState::Active);

    let fault = supervisor.step(SupervisorInput {
        now_ms: 200.0,
        epoch: 1,
        health: Some(healthy(3, 200.0)),
        command: Some(proposal(3, 50.0)),
        emergency_stop: false,
        reset: false,
        shutdown: false,
    });
    assert_eq!(fault.state, SafetyState::Fault);
    assert_eq!(fault.reason, "stale_command");
    assert!(fault.latched);
}

#[test]
fn reset_restarts_the_full_warmup_sequence() {
    let mut supervisor = SafetySupervisor::new(2, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 0.0)).state, SafetyState::Warmup);
    assert_eq!(supervisor.step(input(2, 50.0)).state, SafetyState::Ready);
    assert_eq!(supervisor.step(input(3, 100.0)).state, SafetyState::Active);

    let mut stop = input(4, 150.0);
    stop.emergency_stop = true;
    assert_eq!(supervisor.step(stop).state, SafetyState::EmergencyStop);

    let mut reset = input(6, 250.0);
    reset.reset = true;
    assert_eq!(supervisor.step(reset).state, SafetyState::Warmup);
    assert_eq!(supervisor.step(input(7, 300.0)).state, SafetyState::Ready);
    assert_eq!(supervisor.step(input(8, 350.0)).state, SafetyState::Active);
}

#[test]
fn maximum_frame_id_followed_by_any_frame_faults_without_panicking() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(
        supervisor.step(input(u64::MAX, 10.0)).state,
        SafetyState::Ready
    );

    let fault = supervisor.step(input(0, 20.0));
    assert_eq!(fault.state, SafetyState::Fault);
    assert_eq!(fault.reason, "frame_sequence_gap");
}

#[test]
fn accepted_timestamps_must_not_move_backwards_within_a_session() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 1_000.0)).state, SafetyState::Ready);

    let fault = supervisor.step(input(2, 100.0));
    assert_eq!(fault.state, SafetyState::Fault);
    assert_eq!(fault.reason, "timestamp_regression");
    assert!(fault.latched);
}

#[test]
fn watchdog_boundary_is_inclusive_and_next_representable_value_is_stale() {
    let mut at_boundary = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let accepted = at_boundary.step(SupervisorInput {
        now_ms: 160.0,
        epoch: 1,
        health: Some(healthy(1, 10.0)),
        command: Some(proposal(1, 10.0)),
        emergency_stop: false,
        reset: false,
        shutdown: false,
    });
    assert_eq!(accepted.state, SafetyState::Ready);

    let mut beyond = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let stale = beyond.step(SupervisorInput {
        now_ms: 160.000_001,
        epoch: 1,
        health: Some(healthy(1, 10.0)),
        command: Some(proposal(1, 10.0)),
        emergency_stop: false,
        reset: false,
        shutdown: false,
    });
    assert_eq!(stale.state, SafetyState::Fault);
    assert_eq!(stale.reason, "stale_health");
}

#[test]
fn command_limits_are_inclusive_and_non_finite_values_fail_closed() {
    for (linear, angular) in [(0.0, -0.6), (0.8, 0.6)] {
        let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
        let mut case = input(1, 10.0);
        case.command = Some(ControlProposal {
            linear,
            angular,
            ..proposal(1, 10.0)
        });
        assert_eq!(supervisor.step(case).state, SafetyState::Ready);
    }

    for (linear, angular) in [
        (-f64::EPSILON, 0.0),
        (0.8 + f64::EPSILON, 0.0),
        (0.0, 0.6 + f64::EPSILON),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
        let mut case = input(1, 10.0);
        case.command = Some(ControlProposal {
            linear,
            angular,
            ..proposal(1, 10.0)
        });
        let decision = supervisor.step(case);
        assert_eq!(decision.state, SafetyState::Fault);
        assert_eq!(decision.reason, "invalid_command");
    }
}

#[test]
fn terminal_and_emergency_actions_have_fixed_priority() {
    let mut shutdown_wins = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let mut all_actions = input(1, 10.0);
    all_actions.shutdown = true;
    all_actions.emergency_stop = true;
    all_actions.reset = true;
    assert_eq!(shutdown_wins.step(all_actions).state, SafetyState::Stopped);

    let mut stop_wins = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    let mut stop_and_reset = input(1, 10.0);
    stop_and_reset.emergency_stop = true;
    stop_and_reset.reset = true;
    assert_eq!(
        stop_wins.step(stop_and_reset).state,
        SafetyState::EmergencyStop
    );
}

#[test]
fn equal_timestamps_are_allowed_but_decreasing_timestamps_are_not() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 10.0)).state, SafetyState::Ready);
    assert_eq!(supervisor.step(input(2, 10.0)).state, SafetyState::Active);
}

#[test]
fn first_frame_locks_epoch_and_mismatched_epoch_faults() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 10.0)).state, SafetyState::Ready);

    let mut replay = input(2, 20.0);
    replay.epoch = 999;
    let decision = supervisor.step(replay);
    assert_eq!(decision.state, SafetyState::Fault);
    assert_eq!(decision.reason, "session_mismatch");
}

#[test]
fn new_epoch_with_reset_opens_a_fresh_session() {
    let mut supervisor = SafetySupervisor::new(1, 150.0, 0.8, 0.6).unwrap();
    assert_eq!(supervisor.step(input(1, 10.0)).state, SafetyState::Ready);
    assert_eq!(supervisor.step(input(2, 20.0)).state, SafetyState::Active);

    let mut new_session = input(1, 30.0);
    new_session.epoch = 2;
    new_session.reset = true;
    let decision = supervisor.step(new_session);
    assert_eq!(decision.state, SafetyState::Ready);
}

#[test]
fn constructor_rejects_invalid_configuration() {
    assert!(SafetySupervisor::new(0, 150.0, 0.8, 0.6).is_err());
    assert!(SafetySupervisor::new(1, 0.0, 0.8, 0.6).is_err());
    assert!(SafetySupervisor::new(1, f64::NAN, 0.8, 0.6).is_err());
    assert!(SafetySupervisor::new(1, 150.0, f64::INFINITY, 0.6).is_err());
    assert!(SafetySupervisor::new(1, 150.0, 0.8, f64::NAN).is_err());
    assert!(SafetySupervisor::new(1, 150.0, -0.1, 0.6).is_err());
    assert!(SafetySupervisor::new(1, 150.0, 0.8, -0.1).is_err());
}
