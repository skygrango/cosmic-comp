use std::{collections::VecDeque, num::NonZeroU64, time::Duration};

use smithay::{
    backend::drm::DrmNode,
    utils::{Clock, Monotonic, Time},
};
use tracing::{debug, error};

const BASE_SAFETY_MARGIN: Duration = Duration::from_millis(3);
pub const SAMPLE_TIME_WINDOW: usize = 5;
pub const FULLSCREEN_VRR_PACER_WINDOW: usize = 12;

pub struct Timings {
    pub refresh_interval_ns: Option<NonZeroU64>,
    pub origin_refresh_interval_ns: Option<NonZeroU64>,
    pub vrr_target_rate_internal_ns: Option<NonZeroU64>,
    min_refresh_interval_ns: Option<NonZeroU64>,
    vrr: bool,
    #[allow(dead_code)]
    vendor: Option<u32>,

    pub pending_frame: Option<PendingFrame>,
    pub previous_frames: VecDeque<Frame>,

    // Fullscreen VRR Pacing
    pub last_fullscreen_commit: Option<Time<Monotonic>>,
    pub fullscreen_commit_intervals: VecDeque<Duration>,
    pub avg_fullscreen_commit_interval: Option<Duration>,
    pub last_scheduled_presentation: Option<Duration>,
    pub is_fullscreen: bool,
    pub fullscreen_vrr_fifo: bool,
}

#[derive(Debug)]
pub struct PendingFrame {
    render_start: Time<Monotonic>,
    render_duration_elements: Option<Duration>,
    render_duration_draw: Option<Duration>,
    presentation_submitted: Option<Time<Monotonic>>,
}

#[derive(Debug)]
pub struct Frame {
    pub render_start: Time<Monotonic>,
    pub render_duration_elements: Duration,
    pub render_duration_draw: Duration,
    pub presentation_submitted: Time<Monotonic>,
    pub presentation_presented: Time<Monotonic>,
}

impl Frame {
    fn render_time(&self) -> Duration {
        self.render_duration_elements + self.render_duration_draw
    }

    fn submit_time(&self) -> Duration {
        Time::elapsed(&self.render_start, self.presentation_submitted)
    }

    fn frame_time(&self) -> Duration {
        Time::elapsed(&self.render_start, self.presentation_presented)
    }
}

impl Timings {
    const CLEANUP: usize = 360;

    pub fn new(
        refresh_interval: Option<Duration>,
        min_interval: Option<Duration>,
        vrr: bool,
        node: DrmNode,
    ) -> Self {
        let refresh_interval_ns = if let Some(interval) = &refresh_interval {
            assert_eq!(interval.as_secs(), 0);
            Some(NonZeroU64::new(interval.subsec_nanos().into()).unwrap())
        } else {
            None
        };

        let min_refresh_interval_ns = if let Some(interval) = &min_interval {
            assert_eq!(interval.as_secs(), 0);
            Some(NonZeroU64::new(interval.subsec_nanos().into()).unwrap())
        } else {
            None
        };

        let vendor = if let Ok(vendor) = std::fs::read_to_string(format!(
            "/sys/class/drm/renderD{}/device/vendor",
            node.minor()
        )) {
            u32::from_str_radix(&vendor.trim()[2..], 16).ok()
        } else {
            None
        };

        Self {
            refresh_interval_ns,
            origin_refresh_interval_ns: refresh_interval_ns,
            vrr_target_rate_internal_ns: refresh_interval_ns,
            min_refresh_interval_ns,
            vrr,
            vendor,

            pending_frame: None,
            previous_frames: VecDeque::new(),

            last_fullscreen_commit: None,
            fullscreen_commit_intervals: VecDeque::with_capacity(FULLSCREEN_VRR_PACER_WINDOW),
            avg_fullscreen_commit_interval: None,
            last_scheduled_presentation: None,
            is_fullscreen: false,
            fullscreen_vrr_fifo: false,
        }
    }

    #[cfg(test)]
    pub fn new_for_test(
        refresh_interval: Option<Duration>,
        min_interval: Option<Duration>,
        vrr: bool,
    ) -> Self {
        let refresh_interval_ns = if let Some(interval) = &refresh_interval {
            assert_eq!(interval.as_secs(), 0);
            Some(NonZeroU64::new(interval.subsec_nanos().into()).unwrap())
        } else {
            None
        };

        let min_refresh_interval_ns = if let Some(interval) = &min_interval {
            assert_eq!(interval.as_secs(), 0);
            Some(NonZeroU64::new(interval.subsec_nanos().into()).unwrap())
        } else {
            None
        };

        Self {
            refresh_interval_ns,
            origin_refresh_interval_ns: refresh_interval_ns,
            vrr_target_rate_internal_ns: refresh_interval_ns,
            min_refresh_interval_ns,
            vrr,
            vendor: None,

            pending_frame: None,
            previous_frames: VecDeque::new(),

            last_fullscreen_commit: None,
            fullscreen_commit_intervals: VecDeque::with_capacity(FULLSCREEN_VRR_PACER_WINDOW),
            avg_fullscreen_commit_interval: None,
            last_scheduled_presentation: None,
            is_fullscreen: false,
            fullscreen_vrr_fifo: false,
        }
    }

    pub fn record_fullscreen_commit(&mut self, now: Time<Monotonic>) {
        if let Some(last) = self.last_fullscreen_commit {
            let diff = Time::elapsed(&last, now);
            if diff > Duration::from_millis(250) {
                self.fullscreen_commit_intervals.clear();
                self.avg_fullscreen_commit_interval = None;
            } else if !diff.is_zero() {
                self.fullscreen_commit_intervals.push_back(diff);
                if self.fullscreen_commit_intervals.len() > FULLSCREEN_VRR_PACER_WINDOW {
                    self.fullscreen_commit_intervals.pop_front();
                }
                let sum: Duration = self.fullscreen_commit_intervals.iter().sum();
                self.avg_fullscreen_commit_interval =
                    Some(sum / (self.fullscreen_commit_intervals.len() as u32));
            }
        }
        self.last_fullscreen_commit = Some(now);
    }

    pub fn set_fullscreen_active(&mut self, active: bool) {
        if self.is_fullscreen != active {
            self.is_fullscreen = active;
            self.reset_fullscreen_pacer();
        }
    }

    pub fn reset_fullscreen_pacer(&mut self) {
        self.last_fullscreen_commit = None;
        self.fullscreen_commit_intervals.clear();
        self.avg_fullscreen_commit_interval = None;
        self.last_scheduled_presentation = None;
    }

    pub fn set_fullscreen_vrr_fifo(&mut self, fifo: bool) {
        if self.fullscreen_vrr_fifo != fifo {
            self.fullscreen_vrr_fifo = fifo;
            if !fifo {
                self.reset_fullscreen_pacer();
            }
        }
    }

    pub fn fullscreen_vrr_fifo(&self) -> bool {
        self.fullscreen_vrr_fifo
    }

    pub fn set_avg_fullscreen_commit_interval(&mut self, interval: Option<Duration>) {
        self.avg_fullscreen_commit_interval = interval;
    }

    pub fn avg_fullscreen_commit_interval(&self) -> Option<Duration> {
        self.avg_fullscreen_commit_interval
    }

    pub fn mark_presentation_scheduled(&mut self, target: Duration) {
        self.last_scheduled_presentation = Some(target);
    }

    pub fn refresh_interval(&self) -> Duration {
        match self.refresh_interval_ns {
            Some(ns) => Duration::from_nanos(ns.get()),
            None => Duration::ZERO,
        }
    }

    pub fn set_refresh_interval(&mut self, interval: Option<Duration>) {
        self.refresh_interval_ns = interval
            .map(|duration| duration.subsec_nanos() as u64)
            .and_then(NonZeroU64::new);
        self.origin_refresh_interval_ns = self.refresh_interval_ns;

        self.previous_frames.clear();
    }

    pub fn set_vrr_target_rate_interval(&mut self, interval: Option<Duration>) {
        self.vrr_target_rate_internal_ns = interval
            .map(|duration| duration.subsec_nanos() as u64)
            .and_then(NonZeroU64::new);

        if self.vrr {
            self.refresh_interval_ns = self.vrr_target_rate_internal_ns;
        }
    }

    pub fn set_min_refresh_interval(&mut self, min_interval: Option<Duration>) {
        self.min_refresh_interval_ns = min_interval
            .map(|duration| duration.subsec_nanos() as u64)
            .and_then(NonZeroU64::new);
    }

    pub fn set_vrr(&mut self, vrr: bool) {
        if self.vrr != vrr {
            if vrr {
                self.refresh_interval_ns = self.vrr_target_rate_internal_ns;
            } else {
                self.refresh_interval_ns = self.origin_refresh_interval_ns;
            }
        }
        self.vrr = vrr;
    }

    pub fn vrr(&self) -> bool {
        self.vrr
    }

    pub fn start_render(&mut self, clock: &Clock<Monotonic>) {
        self.pending_frame = Some(PendingFrame {
            render_start: clock.now(),
            render_duration_elements: None,
            render_duration_draw: None,
            presentation_submitted: None,
        });
    }

    pub fn elements_done(&mut self, clock: &Clock<Monotonic>) {
        if let Some(frame) = self.pending_frame.as_mut() {
            frame.render_duration_elements = Some(Time::elapsed(&frame.render_start, clock.now()));
        }
    }

    pub fn draw_done(&mut self, clock: &Clock<Monotonic>) {
        if let Some(frame) = self.pending_frame.as_mut() {
            frame.render_duration_draw = Some(
                Time::elapsed(&frame.render_start, clock.now())
                    - frame.render_duration_elements.unwrap_or(Duration::ZERO),
            );
        }
    }

    pub fn submitted_for_presentation(&mut self, clock: &Clock<Monotonic>) {
        if let Some(frame) = self.pending_frame.as_mut() {
            frame.presentation_submitted = Some(clock.now());
        }
    }

    pub fn presented(&mut self, value: Time<Monotonic>) {
        if let Some(frame) = self.pending_frame.take() {
            let new_frame = Frame {
                render_start: frame.render_start,
                render_duration_elements: frame.render_duration_elements.unwrap_or_default(),
                render_duration_draw: frame.render_duration_draw.unwrap_or_default(),
                presentation_submitted: frame.presentation_submitted.unwrap(),
                presentation_presented: value,
            };
            if new_frame.render_start > new_frame.presentation_submitted {
                debug!(
                    "frame time overflowed: {}",
                    new_frame.frame_time().as_millis()
                );
            }
            self.previous_frames.push_back(new_frame);

            if let Some(overflow) = self.previous_frames.len().checked_sub(Self::CLEANUP * 2) {
                self.previous_frames = self.previous_frames.split_off(overflow + Self::CLEANUP);
            }
        }
    }

    pub fn discard_current_frame(&mut self) {
        let _ = self.pending_frame.take();
    }

    pub fn max_rendertime(&self) -> Duration {
        self.previous_frames
            .iter()
            .map(|f| f.render_time())
            .max()
            .unwrap_or(Duration::ZERO)
    }

    pub fn min_rendertime(&self) -> Duration {
        self.previous_frames
            .iter()
            .map(|f| f.render_time())
            .min()
            .unwrap_or(Duration::ZERO)
    }

    pub fn max_frametime(&self, window: usize) -> Duration {
        self.previous_frames
            .iter()
            .rev()
            .take(window)
            .map(|f| f.frame_time())
            .max()
            .unwrap_or(Duration::ZERO)
    }

    pub fn min_frametime(&self, window: usize) -> Duration {
        self.previous_frames
            .iter()
            .rev()
            .take(window)
            .map(|f| f.frame_time())
            .min()
            .unwrap_or(Duration::ZERO)
    }

    pub fn avg_rendertime(&self) -> Duration {
        let Some(sum_rendertime) = self
            .previous_frames
            .iter()
            .map(|f| f.render_time())
            .try_fold(Duration::ZERO, |acc, x| acc.checked_add(x))
        else {
            return Duration::ZERO;
        };

        sum_rendertime
            .checked_div(self.previous_frames.len() as u32)
            .unwrap_or(Duration::ZERO)
    }

    pub fn avg_submittime(&self, window: usize) -> Option<Duration> {
        if self.previous_frames.len() < window || window == 0 {
            return None;
        }

        Some(
            self.previous_frames
                .iter()
                .rev()
                .take(window)
                .map(|f| f.submit_time())
                .try_fold(Duration::ZERO, |acc, x| acc.checked_add(x))?
                / (window.min(self.previous_frames.len()) as u32),
        )
    }

    pub fn avg_frametime(&self, window: usize) -> Option<Duration> {
        if self.previous_frames.len() < window || window == 0 {
            return None;
        }

        Some(
            self.previous_frames
                .iter()
                .rev()
                .take(window)
                .map(|f| f.frame_time())
                .try_fold(Duration::ZERO, |acc, x| acc.checked_add(x))?
                / (window.min(self.previous_frames.len()) as u32),
        )
    }

    pub fn avg_fps(&self) -> f64 {
        if self.previous_frames.is_empty() {
            return 0.0;
        }
        let secs = match (self.previous_frames.front(), self.previous_frames.back()) {
            (Some(Frame { render_start, .. }), Some(end_frame)) => {
                Time::elapsed(render_start, end_frame.render_start) + end_frame.frame_time()
            }
            _ => {
                return 0.0;
            }
        }
        .as_secs_f64();

        1.0 / (secs / self.previous_frames.len() as f64)
    }

    pub fn next_presentation_time(&self, clock: &Clock<Monotonic>) -> Duration {
        let mut now = clock.now().into();

        let Some(refresh_interval_ns) = self.refresh_interval_ns else {
            return Duration::ZERO;
        };
        let Some(last_presentation_time): Option<Duration> = self
            .previous_frames
            .back()
            .map(|frame| frame.presentation_presented.into())
        else {
            return Duration::ZERO;
        };
        let refresh_interval_ns = refresh_interval_ns.get();

        if self.vrr {
            if self.is_fullscreen
                && self.fullscreen_vrr_fifo
                && let Some(avg_interval) = self.avg_fullscreen_commit_interval
            {
                let refresh_interval = Duration::from_nanos(refresh_interval_ns);
                // Strategy: if average rate is faster than screen refresh rate, use refresh_interval;
                // otherwise use avg_interval.
                let target_interval = if avg_interval < refresh_interval {
                    refresh_interval
                } else {
                    avg_interval
                };

                let base_time = match self.last_scheduled_presentation {
                    Some(scheduled) if scheduled > last_presentation_time && scheduled > now => {
                        scheduled
                    }
                    _ => last_presentation_time.max(now),
                };

                let target_presentation = base_time + target_interval;
                return target_presentation.saturating_sub(now);
            }

            let earliest_presentation =
                last_presentation_time + Duration::from_nanos(refresh_interval_ns);
            return earliest_presentation.saturating_sub(now);
        }

        if now <= last_presentation_time {
            // Got an early VBlank.
            let orig_now = now;
            now += Duration::from_nanos(refresh_interval_ns);

            if now < last_presentation_time {
                // Not sure when this can happen.
                error!(
                    now = ?orig_now,
                    ?last_presentation_time,
                    "got a 2+ early VBlank, {:?} until presentation",
                    last_presentation_time - now,
                );
                now = last_presentation_time + Duration::from_nanos(refresh_interval_ns);
            }
        }

        let since_last = now - last_presentation_time;
        let since_last_ns =
            since_last.as_secs() * 1_000_000_000 + u64::from(since_last.subsec_nanos());
        let to_next_ns = (since_last_ns / refresh_interval_ns + 1) * refresh_interval_ns;

        // If VRR is enabled and more than one frame passed since last presentation, assume that we
        // can present immediately.
        if self.vrr && to_next_ns > refresh_interval_ns {
            Duration::ZERO
        } else {
            last_presentation_time + Duration::from_nanos(to_next_ns) - now
        }
    }

    pub fn past_min_render_time(&self, clock: &Clock<Monotonic>) -> bool {
        let now: Duration = clock.now().into();
        let Some(min_refresh_interval_ns) = self.min_refresh_interval_ns else {
            return true;
        };
        let Some(last_presentation_time): Option<Duration> = self
            .previous_frames
            .back()
            .map(|frame| frame.presentation_presented.into())
        else {
            return true;
        };

        let min_refresh_interval_ns = min_refresh_interval_ns.get();
        if now <= last_presentation_time {
            return false;
        }

        const MIN_MARGIN: Duration = Duration::from_millis(3);
        let baseline = if let Some(refresh_interval_ns) = self.refresh_interval_ns {
            MIN_MARGIN.max(Duration::from_nanos(refresh_interval_ns.get() / 2))
        } else {
            MIN_MARGIN
        };

        let next_presentation_time =
            last_presentation_time + Duration::from_nanos(min_refresh_interval_ns);
        let deadline = next_presentation_time.saturating_sub(
            if let Some(avg_submittime) = self.avg_submittime(SAMPLE_TIME_WINDOW) {
                avg_submittime
            } else {
                baseline
            } + BASE_SAFETY_MARGIN,
        );

        now >= deadline
    }

    pub fn next_render_time(&self, clock: &Clock<Monotonic>) -> Duration {
        let Some(refresh_interval) = self.refresh_interval_ns else {
            return Duration::ZERO; // we don't know what to expect, so render immediately.
        };

        const MIN_MARGIN: Duration = Duration::from_millis(3);
        let baseline = MIN_MARGIN.max(Duration::from_nanos(refresh_interval.get() / 2));

        let estimated_presentation_time = self.next_presentation_time(clock);
        if estimated_presentation_time.is_zero() {
            return Duration::ZERO;
        }

        // HACK: Nvidia returns `page_flip`/`commit` early, so we have no information to optimize latency on submission.
        // if self.vendor == Some(0x10de) {
        //     return Duration::ZERO;
        // }

        let margin = if self.vrr {
            let Some(avg_frametime) = self.avg_frametime(SAMPLE_TIME_WINDOW) else {
                return estimated_presentation_time.saturating_sub(baseline + BASE_SAFETY_MARGIN);
            };
            avg_frametime
        } else {
            let Some(avg_submittime) = self.avg_submittime(SAMPLE_TIME_WINDOW) else {
                return estimated_presentation_time.saturating_sub(baseline + BASE_SAFETY_MARGIN);
            };
            avg_submittime + BASE_SAFETY_MARGIN
        };

        estimated_presentation_time.saturating_sub(margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_timings(
        refresh_interval: Option<Duration>,
        min_interval: Option<Duration>,
        vrr: bool,
    ) -> Timings {
        Timings::new_for_test(refresh_interval, min_interval, vrr)
    }

    #[test]
    fn test_fullscreen_vrr_commit_rolling_average() {
        let mut timings = create_dummy_timings(Some(Duration::from_millis(16)), None, true);
        timings.set_fullscreen_active(true);

        let mut clock_now = Time::from(Duration::from_secs(10));
        timings.record_fullscreen_commit(clock_now);

        // Simulate 2x Frame Generation jitter: alternating 8ms and 24ms intervals
        for i in 0..12 {
            let interval = if i % 2 == 0 {
                Duration::from_millis(8)
            } else {
                Duration::from_millis(24)
            };
            clock_now = Time::from(Duration::from(clock_now) + interval);
            timings.record_fullscreen_commit(clock_now);
        }

        assert_eq!(
            timings.fullscreen_commit_intervals.len(),
            FULLSCREEN_VRR_PACER_WINDOW
        );
        // (8 * 6 + 24 * 6) / 12 = 16ms
        assert_eq!(
            timings.avg_fullscreen_commit_interval,
            Some(Duration::from_millis(16))
        );

        // Record another frame (8ms). The oldest interval was 8ms, so average remains 16ms!
        clock_now = Time::from(Duration::from(clock_now) + Duration::from_millis(8));
        timings.record_fullscreen_commit(clock_now);
        assert_eq!(
            timings.fullscreen_commit_intervals.len(),
            FULLSCREEN_VRR_PACER_WINDOW
        );
        assert_eq!(
            timings.avg_fullscreen_commit_interval,
            Some(Duration::from_millis(16))
        );
    }

    #[test]
    fn test_fullscreen_vrr_pacer_pacing_strategy() {
        let refresh_interval = Duration::from_nanos(6_944_444); // ~144Hz
        let mut timings = create_dummy_timings(Some(refresh_interval), None, true);
        timings.set_fullscreen_active(true);
        timings.set_fullscreen_vrr_fifo(true);

        let clock: Clock<Monotonic> = Clock::new();
        let now_dur: Duration = clock.now().into();

        timings.previous_frames.push_back(Frame {
            render_start: Time::from(now_dur),
            render_duration_elements: Duration::ZERO,
            render_duration_draw: Duration::ZERO,
            presentation_submitted: Time::from(now_dur),
            presentation_presented: Time::from(now_dur),
        });

        // Case A: Commit rate faster than refresh rate (e.g. 4ms interval)
        // Must clamp to refresh_interval (6.944ms)
        timings.avg_fullscreen_commit_interval = Some(Duration::from_millis(4));
        let wait_a = timings.next_presentation_time(&clock);
        // Since now is roughly equal to presentation_presented, wait_a should be approx refresh_interval
        assert!(wait_a <= refresh_interval);
        assert!(wait_a > Duration::from_millis(3));

        // Case B: Commit rate slower than refresh rate (e.g. 16.6ms / 60fps)
        // Must use avg_interval (16.6ms)
        let slow_interval = Duration::from_millis(16);
        timings.avg_fullscreen_commit_interval = Some(slow_interval);
        let wait_b = timings.next_presentation_time(&clock);
        assert!(wait_b <= slow_interval);
        assert!(wait_b > refresh_interval);
    }

    #[test]
    fn test_presentation_time_chaining_burst() {
        let refresh_interval = Duration::from_millis(10);
        let mut timings = create_dummy_timings(Some(refresh_interval), None, true);
        timings.set_fullscreen_active(true);
        timings.set_fullscreen_vrr_fifo(true);
        timings.avg_fullscreen_commit_interval = Some(Duration::from_millis(10));

        let clock: Clock<Monotonic> = Clock::new();
        let now_dur: Duration = clock.now().into();

        timings.previous_frames.push_back(Frame {
            render_start: Time::from(now_dur),
            render_duration_elements: Duration::ZERO,
            render_duration_draw: Duration::ZERO,
            presentation_submitted: Time::from(now_dur),
            presentation_presented: Time::from(now_dur),
        });

        // Schedule first frame at T1 = now_dur + 10ms
        let target1 = now_dur + Duration::from_millis(10);
        timings.mark_presentation_scheduled(target1);

        // Next presentation time should chain from target1, targeting target2 = target1 + 10ms
        let wait_second = timings.next_presentation_time(&clock);
        let current_now: Duration = clock.now().into();
        let expected_target2 = target1 + Duration::from_millis(10);
        let expected_wait = expected_target2.saturating_sub(current_now);

        // Tolerance within 1ms due to clock progression
        let diff = if wait_second > expected_wait {
            wait_second - expected_wait
        } else {
            expected_wait - wait_second
        };
        assert!(diff < Duration::from_millis(1));
    }

    #[test]
    fn test_fullscreen_pacer_reset() {
        let mut timings = create_dummy_timings(Some(Duration::from_millis(16)), None, true);
        timings.set_fullscreen_active(true);
        timings.last_fullscreen_commit = Some(Time::from(Duration::from_secs(1)));
        timings
            .fullscreen_commit_intervals
            .push_back(Duration::from_millis(16));
        timings.avg_fullscreen_commit_interval = Some(Duration::from_millis(16));
        timings.last_scheduled_presentation = Some(Duration::from_secs(2));

        timings.set_fullscreen_active(false);

        assert!(!timings.is_fullscreen);
        assert!(timings.last_fullscreen_commit.is_none());
        assert!(timings.fullscreen_commit_intervals.is_empty());
        assert!(timings.avg_fullscreen_commit_interval.is_none());
        assert!(timings.last_scheduled_presentation.is_none());
    }

    #[test]
    fn test_fullscreen_vrr_fifo_toggle() {
        let clock = Clock::<Monotonic>::new();
        let mut timings = create_dummy_timings(Some(Duration::from_millis(16)), None, true);
        timings.set_fullscreen_active(true);
        timings.avg_fullscreen_commit_interval = Some(Duration::from_millis(20));

        let now_dur: Duration = clock.now().into();
        timings.previous_frames.push_back(Frame {
            render_start: Time::from(now_dur),
            render_duration_elements: Duration::ZERO,
            render_duration_draw: Duration::ZERO,
            presentation_submitted: Time::from(now_dur),
            presentation_presented: Time::from(now_dur),
        });

        // When fullscreen_vrr_fifo is true, pacing strategy uses avg_fullscreen_commit_interval (20ms)
        timings.set_fullscreen_vrr_fifo(true);
        let wait_with_fifo = timings.next_presentation_time(&clock);
        assert!(wait_with_fifo >= Duration::from_millis(19));

        // When fullscreen_vrr_fifo is set to false, it resets pacer and falls back to standard earliest presentation (16ms)
        timings.set_fullscreen_vrr_fifo(false);
        assert!(!timings.fullscreen_vrr_fifo());
        assert!(timings.avg_fullscreen_commit_interval.is_none());

        let wait_without_fifo = timings.next_presentation_time(&clock);
        assert!(wait_without_fifo <= Duration::from_millis(17));
    }
}
