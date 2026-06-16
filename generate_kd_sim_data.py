import json
import os
from datetime import datetime, timezone, timedelta

import numpy as np


FRAME_PERIOD = 0.05
ECG_FS = 500
RX = 3
CHIRPS = 32
SAMPLES = 128


def heart_rate_profile(t, base_hr, variability, event_center=None):
    hr = base_hr + variability * np.sin(2 * np.pi * t / 38.0)
    hr += 3.0 * np.sin(2 * np.pi * t / 11.0 + 0.7)
    if event_center is not None:
        hr += 18.0 * np.exp(-0.5 * ((t - event_center) / 8.0) ** 2)
    return np.clip(hr, 55.0, 145.0)


def integrate_phase(times, hr_bpm):
    freq = hr_bpm / 60.0
    dt = np.diff(times, prepend=times[0])
    return 2 * np.pi * np.cumsum(freq * dt)


def synth_ecg(times, hr_bpm, rng):
    phase = integrate_phase(times, hr_bpm)
    beat_phase = np.mod(phase, 2 * np.pi)
    r = np.exp(-0.5 * (np.minimum(beat_phase, 2 * np.pi - beat_phase) / 0.055) ** 2)
    p = 0.16 * np.exp(-0.5 * ((beat_phase - 1.15) / 0.16) ** 2)
    t_wave = 0.34 * np.exp(-0.5 * ((beat_phase - 4.7) / 0.34) ** 2)
    baseline = 70.0 * np.sin(2 * np.pi * 0.22 * times + 0.3)
    noise = rng.normal(0.0, 22.0, size=times.shape)
    ecg = 1250.0 * r + 180.0 * p + 300.0 * t_wave + baseline + noise
    return np.clip(ecg - np.mean(ecg), -3200, 3200).astype(int)


def synth_radar(n_frames, hr_bpm, rng, motion_level):
    frame_times = np.arange(n_frames) * FRAME_PERIOD
    hr_phase = integrate_phase(frame_times, hr_bpm)
    resp_phase = 2 * np.pi * 0.27 * frame_times
    slow_motion = motion_level * np.sin(2 * np.pi * 0.08 * frame_times + 1.1)
    target_bin = 58 + rng.integers(-4, 5)
    sample_axis = np.arange(SAMPLES)
    range_profile = np.exp(-0.5 * ((sample_axis - target_bin) / 5.5) ** 2)
    static_profile = 0.18 * np.exp(-0.5 * ((sample_axis - 25) / 9.0) ** 2)

    radar = np.empty((n_frames, RX, CHIRPS, SAMPLES), dtype=np.uint16)
    for idx, t in enumerate(frame_times):
        chest_motion = 0.38 * np.sin(hr_phase[idx]) + 0.9 * np.sin(resp_phase[idx]) + slow_motion[idx]
        heart_amp = 1.0 + 0.09 * np.sin(hr_phase[idx]) + 0.03 * np.sin(2 * hr_phase[idx])
        for rx in range(RX):
            rx_gain = 1.0 + 0.04 * rx + rng.normal(0.0, 0.004)
            for chirp in range(CHIRPS):
                chirp_phase = 2 * np.pi * chirp / CHIRPS
                carrier = np.cos(0.18 * sample_axis + 0.12 * chest_motion + chirp_phase)
                target = 260.0 * rx_gain * heart_amp * range_profile * carrier
                clutter = 120.0 * static_profile * np.cos(0.08 * sample_axis + 0.1 * rx)
                thermal = rng.normal(0.0, 24.0 + 10.0 * motion_level, size=SAMPLES)
                value = 1850.0 + target + clutter + thermal
                radar[idx, rx, chirp] = np.clip(value, 0, 4095).astype(np.uint16)
    return radar


def write_ecg(path, start_epoch, duration, base_hr, variability, rng, event_center=None):
    n = int(duration * ECG_FS)
    sample_times = np.arange(n) / ECG_FS
    hr = heart_rate_profile(sample_times, base_hr, variability, event_center)
    ecg = synth_ecg(sample_times, hr, rng)

    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp: ADC HeartRate4sAverage HeartRate30sAverage\n")
        for idx, (relative_t, adc, hr_value) in enumerate(zip(sample_times, ecg, hr)):
            group_t = np.floor(relative_t / 0.032) * 0.032
            timestamp = start_epoch + group_t
            hr4 = int(round(hr_value))
            if idx < ECG_FS * 2:
                hr30 = hr4
            else:
                left = max(0, idx - ECG_FS * 30)
                hr30 = int(round(float(np.mean(hr[left:idx + 1]))))
            f.write(f"{timestamp:.3f}: {int(adc):6d} {hr4:3d} {hr30:3d}\n")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def generate_session(root, session_idx, base_hr, variability, duration, motion_level, event_center=None):
    rng = np.random.default_rng(20260616 + session_idx)
    start_dt = datetime(2026, 6, 16, 9 + session_idx, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    start_epoch = start_dt.timestamp()
    record_name = f"BGT60TR13C_record_{start_dt.strftime('%Y%m%d-%H%M%S')}_sim{session_idx}"
    record_dir = os.path.join(root, record_name)
    radar_dir = os.path.join(record_dir, "RadarIfxAvian_00")
    os.makedirs(radar_dir, exist_ok=True)

    n_frames = int(duration / FRAME_PERIOD)
    frame_times = np.arange(n_frames) * FRAME_PERIOD
    frame_hr = heart_rate_profile(frame_times, base_hr, variability, event_center)
    radar = synth_radar(n_frames, frame_hr, rng, motion_level)
    np.save(os.path.join(radar_dir, "radar.npy"), radar)

    config = {
        "device_config": {
            "fmcw_single_shape": {
                "aaf_cutoff_Hz": 500000,
                "chirp_repetition_time_s": 0.004999999888241291,
                "end_frequency_Hz": 61500000000,
                "frame_repetition_time_s": FRAME_PERIOD,
                "hp_cutoff_Hz": 80000,
                "if_gain_dB": 33,
                "mimo_mode": "off",
                "num_chirps_per_frame": CHIRPS,
                "num_samples_per_chirp": SAMPLES,
                "rx_antennas": [1, 2, 3],
                "sample_rate_Hz": 2000000,
                "start_frequency_Hz": 59500000000,
                "tx_antennas": [1],
                "tx_power_level": 31,
            }
        }
    }
    write_json(os.path.join(radar_dir, "config.json"), config)
    write_json(os.path.join(radar_dir, "meta.json"), {
        "adc_resolution": 12,
        "description": "Simulated BGT60TR13C FMCW Radar Sensor",
        "sdk_version": "simulated",
        "uuid": f"sim-{session_idx}",
    })
    write_json(os.path.join(record_dir, "meta.json"), {
        "date_captured": start_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": "simulated",
        "ifxdaq_version": "simulated",
        "machine": "AMD64",
        "os": "Windows",
        "username": "simulated",
    })

    ecg_path = os.path.join(root, f"ECGLog-sim-{session_idx}-{start_dt.strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    write_ecg(ecg_path, start_epoch, duration, base_hr, variability, rng, event_center)
    return os.path.join(radar_dir, "radar.npy"), ecg_path


def main():
    root = os.path.join(os.getcwd(), "simulated_kd_data")
    os.makedirs(root, exist_ok=True)
    sessions = [
        (1, 68.0, 5.0, 75.0, 0.25, None),
        (2, 92.0, 8.0, 80.0, 0.45, 44.0),
        (3, 118.0, 10.0, 78.0, 0.65, 38.0),
    ]
    print("Generated simulated KD training data:")
    for args in sessions:
        radar_path, ecg_path = generate_session(root, *args)
        print(f"  radar: {radar_path}")
        print(f"  ecg:   {ecg_path}")


if __name__ == "__main__":
    main()
