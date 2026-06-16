import time

import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp


CONFIG = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.05,
    chirp_repetition_time_s=500e-6,
    num_chirps=16,
    tdm_mimo=False,
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59e9,
        end_frequency_Hz=61e9,
        sample_rate_Hz=2e6,
        num_samples=128,
        rx_mask=7,
        tx_mask=1,
        tx_power_level=31,
        lp_cutoff_Hz=500000,
        hp_cutoff_Hz=80000,
        if_gain_dB=30,
    ),
)


def range_axis(num_samples):
    bandwidth = CONFIG.chirp.end_frequency_Hz - CONFIG.chirp.start_frequency_Hz
    return np.arange(num_samples // 2) * 3e8 / (2.0 * bandwidth)


def analyze_frame(frame, target_min_m=0.20, target_max_m=1.20):
    frame = np.asarray(frame)
    centered = frame - np.mean(frame, axis=2, keepdims=True)
    window = np.hanning(frame.shape[2]).reshape(1, 1, -1)
    range_fft = np.fft.fft(centered * window, axis=2)[..., : frame.shape[2] // 2]
    magnitude = np.mean(np.abs(range_fft), axis=(0, 1))
    ranges = range_axis(frame.shape[2])
    roi = np.where((ranges >= target_min_m) & (ranges <= target_max_m))[0]
    if roi.size == 0:
        roi = np.arange(2, len(magnitude))
    roi_mag = magnitude[roi]
    peak_idx = int(roi[np.argmax(roi_mag)])
    noise = float(np.median(roi_mag) + 1e-6)
    peak = float(magnitude[peak_idx])
    return peak_idx, float(ranges[peak_idx]), peak / noise, peak


def main():
    with DeviceFmcw() as device:
        print("Radar SDK:", get_version())
        print("UUID:", device.get_board_uuid())
        print("Sensor:", device.get_sensor_type())
        sequence = device.create_simple_sequence(CONFIG)
        device.set_acquisition_sequence(sequence)
        print("Reading frames. Put chest ~30 cm from radar, then move away.")
        for idx in range(120):
            frames = device.get_next_frame()
            for frame in frames:
                peak_bin, distance_m, snr, peak = analyze_frame(frame)
                print(f"{idx:03d} shape={frame.shape} peak_bin={peak_bin:02d} distance={distance_m:.2f}m snr={snr:.2f} peak={peak:.1f}")
            time.sleep(0.02)


if __name__ == "__main__":
    main()
