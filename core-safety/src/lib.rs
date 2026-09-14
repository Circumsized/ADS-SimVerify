#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyState {
    Boot,
    Warmup,
    Ready,
    Active,
    EmergencyStop,
    Fault,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeHealth {
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub sensor_valid: bool,
    pub perception_valid: bool,
    pub solver_valid: bool,
    pub articulation_ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlProposal {
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub linear: f64,
    pub angular: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupervisorInput {
    pub now_ms: f64,
    pub epoch: u64,
    pub health: Option<RuntimeHealth>,
    pub command: Option<ControlProposal>,
    pub emergency_stop: bool,
    pub reset: bool,
    pub shutdown: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafetyDecision {
    pub state: SafetyState,
    pub linear: f64,
    pub angular: f64,
    pub reason: String,
    pub latched: bool,
}

pub struct SafetySupervisor {
    warmup_frames: u64,
    watchdog_ms: f64,
    max_linear: f64,
    max_angular: f64,
    state: SafetyState,
    reason: String,
    healthy_frames: u64,
    last_frame_id: Option<u64>,
    last_now_ms: Option<f64>,
    last_timestamp_ms: Option<f64>,
    last_epoch: Option<u64>,
}

impl SafetySupervisor {
    pub fn new(
        warmup_frames: u64,
        watchdog_ms: f64,
        max_linear: f64,
        max_angular: f64,
    ) -> Result<Self, String> {
        if warmup_frames < 1 || !watchdog_ms.is_finite() || watchdog_ms <= 0.0 {
            return Err("warmup_frames and watchdog_ms must be positive".to_string());
        }
        if !max_linear.is_finite()
            || max_linear < 0.0
            || !max_angular.is_finite()
            || max_angular < 0.0
        {
            return Err("command limits must be finite and non-negative".to_string());
        }
        Ok(Self {
            warmup_frames,
            watchdog_ms,
            max_linear,
            max_angular,
            state: SafetyState::Boot,
            reason: "boot".to_string(),
            healthy_frames: 0,
            last_frame_id: None,
            last_now_ms: None,
            last_timestamp_ms: None,
            last_epoch: None,
        })
    }

    fn zero(&self) -> SafetyDecision {
        SafetyDecision {
            state: self.state,
            linear: 0.0,
            angular: 0.0,
            reason: self.reason.clone(),
            latched: matches!(self.state, SafetyState::EmergencyStop | SafetyState::Fault),
        }
    }

    fn fault(&mut self, reason: &str) -> SafetyDecision {
        self.state = SafetyState::Fault;
        self.reason = reason.to_string();
        self.zero()
    }

    fn validate(&self, input: &SupervisorInput) -> Option<&'static str> {
        let (Some(health), Some(command)) = (input.health.as_ref(), input.command.as_ref()) else {
            return Some("missing_runtime_input");
        };
        if !(health.sensor_valid
            && health.perception_valid
            && health.solver_valid
            && health.articulation_ready)
        {
            return Some("runtime_unhealthy");
        }
        if health.frame_id != command.frame_id {
            return Some("frame_mismatch");
        }
        if self
            .last_frame_id
            .is_some_and(|last| last.checked_add(1) != Some(health.frame_id))
        {
            return Some("frame_sequence_gap");
        }
        if !input.now_ms.is_finite()
            || !health.timestamp_ms.is_finite()
            || input.now_ms < health.timestamp_ms
        {
            return Some("invalid_health_timestamp");
        }
        if self.last_now_ms.is_some_and(|last| input.now_ms < last)
            || self
                .last_timestamp_ms
                .is_some_and(|last| health.timestamp_ms < last)
        {
            return Some("timestamp_regression");
        }
        if input.now_ms - health.timestamp_ms > self.watchdog_ms {
            return Some("stale_health");
        }
        if !command.timestamp_ms.is_finite() || input.now_ms < command.timestamp_ms {
            return Some("invalid_command_timestamp");
        }
        if input.now_ms - command.timestamp_ms > self.watchdog_ms {
            return Some("stale_command");
        }
        if health.timestamp_ms != command.timestamp_ms {
            return Some("timestamp_mismatch");
        }
        if !command.linear.is_finite()
            || !command.angular.is_finite()
            || !(0.0..=self.max_linear).contains(&command.linear)
            || command.angular.abs() > self.max_angular
        {
            return Some("invalid_command");
        }
        None
    }

    pub fn step(&mut self, input: SupervisorInput) -> SafetyDecision {
        if self.state == SafetyState::Stopped {
            return self.zero();
        }
        if input.shutdown {
            self.state = SafetyState::Stopped;
            self.reason = "shutdown".to_string();
            return self.zero();
        }
        if input.emergency_stop {
            self.state = SafetyState::EmergencyStop;
            self.reason = "emergency_stop".to_string();
            return self.zero();
        }
        if matches!(self.state, SafetyState::EmergencyStop | SafetyState::Fault) {
            if !input.reset {
                return self.zero();
            }
            self.state = SafetyState::Boot;
            self.reason = "reset".to_string();
            self.healthy_frames = 0;
            self.last_frame_id = None;
            self.last_now_ms = None;
            self.last_timestamp_ms = None;
        }

        match self.last_epoch {
            None => self.last_epoch = Some(input.epoch),
            Some(last) if last != input.epoch => {
                if !input.reset {
                    return self.fault("session_mismatch");
                }
                self.state = SafetyState::Boot;
                self.reason = "reset".to_string();
                self.healthy_frames = 0;
                self.last_frame_id = None;
                self.last_now_ms = None;
                self.last_timestamp_ms = None;
                self.last_epoch = Some(input.epoch);
            }
            Some(_) => {}
        }

        if let Some(reason) = self.validate(&input) {
            return self.fault(reason);
        }
        let health = input.health.expect("validated health is present");
        let command = input.command.expect("validated command is present");
        self.last_frame_id = Some(health.frame_id);
        self.last_now_ms = Some(input.now_ms);
        self.last_timestamp_ms = Some(health.timestamp_ms);

        if matches!(self.state, SafetyState::Boot | SafetyState::Warmup) {
            self.healthy_frames += 1;
            if self.healthy_frames < self.warmup_frames {
                self.state = SafetyState::Warmup;
                self.reason = "warming_up".to_string();
            } else {
                self.state = SafetyState::Ready;
                self.reason = "startup_ready".to_string();
            }
            return self.zero();
        }
        if self.state == SafetyState::Ready {
            self.state = SafetyState::Active;
        }
        self.reason = "allow".to_string();
        SafetyDecision {
            state: self.state,
            linear: command.linear,
            angular: command.angular,
            reason: self.reason.clone(),
            latched: false,
        }
    }
}
