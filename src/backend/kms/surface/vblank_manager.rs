use std::{
    sync::{Arc, Mutex, Condvar, atomic::{AtomicBool, AtomicU64, Ordering}},
    thread,
};
use smithay::utils::{Clock, Monotonic};

// Default constants based on gamescope
const STARTING_VBLANK_DRAW_TIME: u64 = 3_000_000; // 3ms
const DEFAULT_VBLANK_RED_ZONE: u64 = 1_650_000; // 1.65ms
const DEFAULT_VBLANK_DRAW_TIME_MIN_COMPOSITING: u64 = 2_400_000; // 2.4ms
const DEFAULT_VBLANK_RATE_OF_DECAY_PERCENTAGE: u64 = 980; // 98%
const VBLANK_RATE_OF_DECAY_MAX: u64 = 1000; // 100%
const VRR_FLUSHING_TIME: u64 = 300_000; // 0.3ms

#[derive(Debug, Clone, Copy)]
pub struct VBlankScheduleTime {
    pub target_vblank: std::time::Duration,
    pub scheduled_wakeup_point: std::time::Duration,
}

pub struct VBlankManager {
    shared: Arc<SharedState>,
    nudge_thread: Option<thread::JoinHandle<()>>,
}

struct SharedState {
    armed: AtomicBool,
    running: AtomicBool,
    condvar: Condvar,
    mutex: Mutex<SharedInner>,

    // Tunables and state
    last_vblank: AtomicU64, // nanos
    last_draw_time: AtomicU64, // nanos
    currently_compositing: AtomicBool,
}

struct SharedInner {
    refresh_cycle_ns: u64,
    vrr_active: bool,
    preemptive: bool,
    deadline: Option<VBlankScheduleTime>,
    rolling_max_draw_time: u64,
    vblank_draw_buffer_red_zone: u64,
    vblank_draw_time_min_compositing: u64,
    vblank_rate_of_decay_percentage: u64,
}

impl VBlankManager {
    pub fn new() -> (Self, calloop::channel::Channel<VBlankScheduleTime>) {
        let (sender, receiver) = calloop::channel::channel();

        let clock: Clock<Monotonic> = Clock::new();
        let now: std::time::Duration = clock.now().into();

        let shared = Arc::new(SharedState {
            armed: AtomicBool::new(false),
            running: AtomicBool::new(true),
            condvar: Condvar::new(),
            mutex: Mutex::new(SharedInner {
                refresh_cycle_ns: 16_666_666, // Default 60Hz
                vrr_active: false,
                preemptive: false,
                deadline: None,
                rolling_max_draw_time: STARTING_VBLANK_DRAW_TIME,
                vblank_draw_buffer_red_zone: DEFAULT_VBLANK_RED_ZONE,
                vblank_draw_time_min_compositing: DEFAULT_VBLANK_DRAW_TIME_MIN_COMPOSITING,
                vblank_rate_of_decay_percentage: DEFAULT_VBLANK_RATE_OF_DECAY_PERCENTAGE,
            }),
            last_vblank: AtomicU64::new(now.as_nanos() as u64),
            last_draw_time: AtomicU64::new(STARTING_VBLANK_DRAW_TIME),
            currently_compositing: AtomicBool::new(false),
        });

        let shared_clone = shared.clone();

        let nudge_thread = thread::Builder::new()
            .name("cosmic-vblk".into())
            .spawn(move || Self::nudge_thread(shared_clone, sender))
            .unwrap();

        (
            Self {
                shared,
                nudge_thread: Some(nudge_thread),
            },
            receiver,
        )
    }

    pub fn update_refresh_cycle(&self, cycle_ns: u64, vrr_active: bool) {
        let mut inner = self.shared.mutex.lock().unwrap();
        inner.refresh_cycle_ns = cycle_ns;
        inner.vrr_active = vrr_active;
    }

    pub fn mark_vblank(&self, nanos: u64, re_arm: bool) {
        self.shared.last_vblank.store(nanos, Ordering::Relaxed);
        if re_arm {
            self.arm(None, true);
        }
    }

    pub fn update_last_draw_time(&self, nanos: u64) {
        self.shared.last_draw_time.store(nanos, Ordering::Relaxed);
    }

    pub fn update_compositing(&self, compositing: bool) {
        self.shared.currently_compositing.store(compositing, Ordering::Relaxed);
    }

    pub fn arm(&self, schedule: Option<VBlankScheduleTime>, preemptive: bool) {
        let mut inner = self.shared.mutex.lock().unwrap();
        inner.deadline = schedule;
        inner.preemptive = preemptive;
        self.shared.armed.store(true, Ordering::Release);
        self.shared.condvar.notify_all();
    }

    fn calc_next_wakeup_time(
        shared: &SharedState,
        inner: &mut SharedInner,
        now_ns: u64,
    ) -> VBlankScheduleTime {
        if let Some(deadline) = inner.deadline.take() {
            return deadline;
        }

        let vrr = inner.vrr_active;
        let offset;

        if !vrr {
            let red_zone = inner.vblank_draw_buffer_red_zone;
            let decay_alpha = inner.vblank_rate_of_decay_percentage;

            let mut draw_time = shared.last_draw_time.load(Ordering::Relaxed);
            if shared.currently_compositing.load(Ordering::Relaxed) {
                draw_time = draw_time.max(inner.vblank_draw_time_min_compositing);
            }

            let new_rolling_draw_time;
            if (draw_time as i64) - ((red_zone / 2) as i64) > (inner.rolling_max_draw_time as i64) {
                new_rolling_draw_time = draw_time;
            } else {
                new_rolling_draw_time = ((decay_alpha * inner.rolling_max_draw_time)
                    + (VBLANK_RATE_OF_DECAY_MAX - decay_alpha) * draw_time)
                    / VBLANK_RATE_OF_DECAY_MAX;
            }

            let refresh_interval = inner.refresh_cycle_ns;
            let new_rolling_draw_time = new_rolling_draw_time.min(refresh_interval.saturating_sub(red_zone));

            if !inner.preemptive {
                inner.rolling_max_draw_time = new_rolling_draw_time;
            }

            offset = new_rolling_draw_time + red_zone;
        } else {
            if !inner.preemptive {
                inner.rolling_max_draw_time = STARTING_VBLANK_DRAW_TIME;
            }

            let red_zone = VRR_FLUSHING_TIME;
            let mut draw_time = 0;
            if shared.currently_compositing.load(Ordering::Relaxed) {
                draw_time = draw_time.max(inner.vblank_draw_time_min_compositing);
            }

            offset = draw_time + red_zone;
        }

        let last_vblank = shared.last_vblank.load(Ordering::Relaxed);
        let interval = inner.refresh_cycle_ns;

        let mut target_point = last_vblank + interval.saturating_sub(offset);
        while target_point < now_ns {
            target_point += interval;
        }

        let scheduled_wakeup_point = target_point;
        let target_vblank = scheduled_wakeup_point + offset;

        VBlankScheduleTime {
            target_vblank: std::time::Duration::from_nanos(target_vblank),
            scheduled_wakeup_point: std::time::Duration::from_nanos(scheduled_wakeup_point),
        }
    }

    fn nudge_thread(shared: Arc<SharedState>, sender: calloop::channel::Sender<VBlankScheduleTime>) {
        unsafe {
            let min_priority = libc::sched_get_priority_min(libc::SCHED_RR);
            let sp = libc::sched_param {
                sched_priority: min_priority,
            };
            if libc::pthread_setschedparam(
                libc::pthread_self(),
                libc::SCHED_RR | libc::SCHED_RESET_ON_FORK,
                &sp,
            ) != 0
            {
                tracing::warn!("Failed to gain real time thread priority (Check CAP_SYS_NICE)");
            }
        }

        let clock: Clock<Monotonic> = Clock::new();

        loop {
            let mut inner = shared.mutex.lock().unwrap();
            while !shared.armed.load(Ordering::Acquire) && shared.running.load(Ordering::Acquire) {
                inner = shared.condvar.wait(inner).unwrap();
            }

            if !shared.running.load(Ordering::Acquire) {
                break;
            }

            let now: std::time::Duration = clock.now().into();
            let schedule = Self::calc_next_wakeup_time(&shared, &mut inner, now.as_nanos() as u64);

            if schedule.scheduled_wakeup_point <= now || inner.preemptive {
                shared.armed.store(false, Ordering::Release);
                inner.preemptive = false;
                drop(inner);
                if sender.send(schedule).is_err() {
                    break;
                }
                continue;
            }

            let timeout = schedule.scheduled_wakeup_point - now;
            let (inner_after_wait, result) = shared.condvar.wait_timeout(inner, timeout).unwrap();
            inner = inner_after_wait;

            if result.timed_out() {
                shared.armed.store(false, Ordering::Release);
                inner.preemptive = false;
                drop(inner);
                if sender.send(schedule).is_err() {
                    break;
                }
            }
        }
    }
}

impl Drop for VBlankManager {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Release);
        self.shared.armed.store(true, Ordering::Release);
        self.shared.condvar.notify_all();
        if let Some(thread) = self.nudge_thread.take() {
            let _ = thread.join();
        }
    }
}
