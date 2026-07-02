import math
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS

try:
    from scipy.signal import butter, filtfilt
    _has_scipy = True
except Exception:
    butter = None
    filtfilt = None
    _has_scipy = False


app = Flask(__name__)
CORS(app)
app.config["PROPAGATE_EXCEPTIONS"] = True


def zscore(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)


def bandpass(values, fs, low, high, order=3):
    arr = np.asarray(values, dtype=float)
    if arr.size < max(12, int(fs * 2)) or not _has_scipy:
        return arr - np.mean(arr)
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, arr)


class SimulatedVitalMonitor:
    """Realtime simulated monitor for report/demo recording.

    The API mirrors heart_rate_monitor.py, but no radar SDK or BMD101 serial
    device is opened. The noisy radar heart waveform intentionally has lower
    quality, while the KD waveform is a cleaner version synchronized with ECG.
    """

    def __init__(self):
        self.fs = 20.0
        self.ecg_fs = 250.0
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
        self.start_time = time.time()
        self.sim_t = 0.0
        self.frame_count = 0
        self.error_message = ""
        self.rng = np.random.default_rng(20260622)

        self.heart_rate_buffer = deque(maxlen=180)
        self.kd_wave_buffer = deque(maxlen=180)
        self.respiratory_buffer = deque(maxlen=180)
        self.raw_data_buffer = deque(maxlen=180)
        self.ecg_buffer = deque(maxlen=900)
        self.ecg_time_buffer = deque(maxlen=900)

        self.current_heart_rate = 72.0
        self.current_kd_hr = 72.0
        self.current_respiratory_rate = 16.0
        self.ecg_hr = 72.0
        self.kd_status = "simulation_kd"

        self.target_present = True
        self.last_target_distance_m = 0.31
        self.last_target_snr = 18.0
        self.latest_range_doppler = self._make_range_doppler(0.0)
        self.range_axis = [round(x, 3) for x in np.linspace(0.0, 1.6, 64)]
        self.doppler_axis = [round(x, 2) for x in np.linspace(-10, 10, 32)]

        self.beats = []
        self.rr_values = []
        self.next_beat_time = 0.55
        self.prev_rr = 60.0 / 72.0
        self._ensure_beats_until(30.0)

    def _hr_target(self, t):
        return 76.0 + 3.0 * math.sin(2 * math.pi * t / 36.0) + 1.4 * math.sin(2 * math.pi * t / 11.5 + 0.8)

    def _ensure_beats_until(self, t_end):
        while self.next_beat_time < t_end + 3.0:
            target_rr = 60.0 / max(55.0, min(120.0, self._hr_target(self.next_beat_time)))
            jitter = self.rng.normal(0.0, 0.026)
            rr = 0.72 * self.prev_rr + 0.28 * target_rr + jitter
            rr = float(np.clip(rr, 0.55, 1.08))
            self.next_beat_time += rr
            self.prev_rr = rr
            self.beats.append(self.next_beat_time)
            self.rr_values.append(rr)

    def _interp_hr(self, t):
        self._ensure_beats_until(t)
        beats = np.asarray(self.beats, dtype=float)
        rr = np.asarray(self.rr_values, dtype=float)
        if beats.size < 2:
            return self._hr_target(t)
        hr_at_beats = 60.0 / np.clip(rr, 1e-6, None)
        return float(np.interp(t, beats, hr_at_beats, left=hr_at_beats[0], right=hr_at_beats[-1]))

    def _mechanical_heart(self, t):
        self._ensure_beats_until(t)
        value = 0.0
        for i, beat in enumerate(reversed(self.beats[-8:])):
            beat_index = len(self.beats) - 1 - i
            delay = 0.20 + 0.018 * math.sin(beat_index * 0.7)
            dt = t - (beat + delay)
            if -0.05 <= dt <= 0.75:
                rise = 0.035 + 0.004 * math.sin(beat_index)
                decay = 0.22 + 0.025 * math.sin(beat_index * 0.47)
                amp = 1.0 + 0.10 * math.sin(beat_index * 0.39)
                pulse = 0.0
                if dt >= 0:
                    pulse = math.exp(-dt / decay) - math.exp(-dt / rise)
                rebound = -0.20 * math.exp(-0.5 * ((dt - 0.36) / 0.085) ** 2)
                value += amp * (2.0 * pulse + rebound)
        return value

    def _ecg_value(self, t):
        self._ensure_beats_until(t)
        value = 0.0
        for i, beat in enumerate(reversed(self.beats[-6:])):
            beat_index = len(self.beats) - 1 - i
            amp = 1.0 + 0.05 * math.sin(beat_index * 0.43)
            value += 0.16 * amp * math.exp(-0.5 * ((t - (beat - 0.17)) / 0.045) ** 2)
            value += -0.42 * amp * math.exp(-0.5 * ((t - (beat - 0.026)) / 0.012) ** 2)
            value += 2.35 * amp * math.exp(-0.5 * ((t - beat) / 0.014) ** 2)
            value += -0.70 * amp * math.exp(-0.5 * ((t - (beat + 0.030)) / 0.017) ** 2)
            value += 0.38 * amp * math.exp(-0.5 * ((t - (beat + 0.27)) / 0.080) ** 2)
        baseline = 0.08 * math.sin(2 * math.pi * 0.19 * t + 0.4)
        noise = self.rng.normal(0.0, 0.028)
        return 1050.0 * (value + baseline + noise)

    def _make_range_doppler(self, t):
        rows, cols = 32, 64
        y = np.linspace(-1, 1, rows)[:, None]
        x = np.linspace(0, 1.6, cols)[None, :]
        target_range = 0.32 + 0.018 * math.sin(2 * math.pi * 0.22 * t)
        target_doppler = 0.10 * math.sin(2 * math.pi * 1.25 * t)
        target = 38.0 * np.exp(-((x - target_range) ** 2) / 0.006 - ((y - target_doppler) ** 2) / 0.065)
        clutter = 9.0 * np.exp(-((x - 0.16) ** 2) / 0.02 - (y ** 2) / 0.40)
        background = -35.0 + 2.0 * self.rng.normal(size=(rows, cols))
        return (background + clutter + target).tolist()

    def _append_samples(self):
        t = self.sim_t
        hr = self._interp_hr(t)
        resp_rate = 15.8 + 1.2 * math.sin(2 * math.pi * t / 42.0 + 0.3)

        mech = self._mechanical_heart(t)
        resp = 0.75 * math.sin(2 * math.pi * (resp_rate / 60.0) * t + 0.8)
        motion = 0.18 * math.sin(2 * math.pi * 0.07 * t)
        radar_clean = mech + 0.35 * resp + motion

        # Lower quality base radar waveform: multipath drift + bursty noise.
        burst = 0.0
        if int(t * 10) % 47 in (0, 1):
            burst = self.rng.normal(0.0, 0.45)
        noisy_heart = radar_clean + 0.38 * self.rng.normal() + 0.22 * math.sin(2 * math.pi * 2.7 * t) + burst

        # KD waveform: clean, morphology-preserving, aligned with ECG mechanical delay.
        kd = 1.25 * mech + 0.08 * resp + 0.045 * self.rng.normal()
        raw_display = 48.0 + 2.4 * resp + 1.2 * mech + self.rng.normal(0.0, 1.0)

        with self.lock:
            self.frame_count += 1
            self.current_heart_rate = 0.82 * self.current_heart_rate + 0.18 * (hr + self.rng.normal(0.0, 1.8))
            self.current_kd_hr = 0.90 * self.current_kd_hr + 0.10 * (hr + self.rng.normal(0.0, 0.4))
            self.ecg_hr = 0.88 * self.ecg_hr + 0.12 * (hr + self.rng.normal(0.0, 0.3))
            self.current_respiratory_rate = 0.92 * self.current_respiratory_rate + 0.08 * resp_rate
            self.last_target_snr = 17.0 + 2.0 * math.sin(2 * math.pi * t / 15.0) + self.rng.normal(0.0, 0.4)
            self.last_target_distance_m = 0.31 + 0.012 * math.sin(2 * math.pi * 0.22 * t)

            self.heart_rate_buffer.append(float(noisy_heart))
            self.kd_wave_buffer.append(float(kd))
            self.respiratory_buffer.append(float(resp + 0.05 * self.rng.normal()))
            self.raw_data_buffer.append(float(raw_display))
            if self.frame_count % 5 == 0:
                self.latest_range_doppler = self._make_range_doppler(t)

        # ECG is sampled faster than radar, but returned in the same API payload.
        ecg_step = 1.0 / self.ecg_fs
        n_ecg = max(1, int(round(self.ecg_fs / self.fs)))
        for k in range(n_ecg):
            te = t + k * ecg_step
            with self.lock:
                self.ecg_buffer.append(float(self._ecg_value(te)))
                self.ecg_time_buffer.append(float(te))

        self.sim_t += 1.0 / self.fs

    def _loop(self):
        next_tick = time.time()
        while self.is_running:
            self._append_samples()
            next_tick += 1.0 / self.fs
            time.sleep(max(0.0, next_tick - time.time()))

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def _display_wave(self, values, max_points=160):
        arr = np.asarray(list(values)[-max_points:], dtype=float)
        if arr.size == 0:
            return []
        return [float(x) for x in zscore(arr)]

    def get_data(self):
        with self.lock:
            ecg_arr = np.asarray(list(self.ecg_buffer)[-600:], dtype=float)
            ecg_display = zscore(ecg_arr)[-500:].tolist() if ecg_arr.size else []
            heart_wave = self._display_wave(self.heart_rate_buffer)
            kd_wave = self._display_wave(self.kd_wave_buffer)
            resp_wave = self._display_wave(self.respiratory_buffer)
            raw_wave = [float(x) for x in list(self.raw_data_buffer)[-120:]]

            return {
                "heart_rate_waveform": heart_wave,
                "respiratory_waveform": resp_wave,
                "raw_data_waveform": raw_wave,
                "ecg_waveform": [float(x) for x in list(self.ecg_buffer)[-500:]],
                "ecg_display": [float(x) for x in ecg_display],
                "kd_waveform": kd_wave,
                "kd_hr": float(self.current_kd_hr),
                "kd_status": "simulation_kd_clean",
                "target_present": True,
                "target_distance_m": float(self.last_target_distance_m),
                "target_snr": float(max(0.0, self.last_target_snr)),
                "range_doppler_map": self.latest_range_doppler,
                "range_axis": self.range_axis,
                "doppler_axis": self.doppler_axis,
                "current_heart_rate": float(self.current_heart_rate),
                "current_respiratory_rate": float(self.current_respiratory_rate),
                "frame_count": int(self.frame_count),
                "is_running": bool(self.is_running),
                "error_message": self.error_message,
                "buffer_sizes": {
                    "raw_phase": len(self.heart_rate_buffer),
                    "heart_rate": len(self.heart_rate_buffer),
                    "kd": len(self.kd_wave_buffer),
                    "respiratory": len(self.respiratory_buffer),
                    "raw_data": len(self.raw_data_buffer),
                    "ecg": len(self.ecg_buffer),
                },
                "ecg_hr": float(self.ecg_hr),
                "timestamp": float(time.time()),
                "simulation_mode": True,
            }


processor = SimulatedVitalMonitor()


@app.route("/")
def index():
    return render_template("index_sim.html")


@app.route("/mobile")
@app.route("/m")
def mobile_index():
    return render_template("mobile.html")


@app.route("/api/data")
def get_data():
    return jsonify(processor.get_data())


@app.route("/api/start")
def start_monitoring():
    processor.start()
    return jsonify({"status": "started", "simulation": True})


@app.route("/api/stop")
def stop_monitoring():
    processor.stop()
    return jsonify({"status": "stopped", "simulation": True})


if __name__ == "__main__":
    print("=" * 60)
    print("Simulated radar / ECG vital-sign monitor")
    print("No radar SDK or BMD101 serial connection will be opened.")
    print("=" * 60)
    processor.start()
    print("Ready: http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
