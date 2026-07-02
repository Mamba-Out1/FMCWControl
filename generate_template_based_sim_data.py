import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TEMPLATE_RADAR_DIR = ROOT / "bgtr13cTest" / "BGT60TR13C_record_20260604-161236" / "RadarIfxAvian_00"
TEMPLATE_RECORD_DIR = TEMPLATE_RADAR_DIR.parent
ECG_REFERENCE = ROOT / "ECGLog-2026-5-7-15-40-57.txt"
OUT_ROOT = ROOT / "simulated_kd_data"

ECG_FS = 500
DURATION_S = 40.0
SESSIONS = [
	{
		"idx": 1,
		"start": datetime(2026, 6, 17, 18, 0, 0, tzinfo=timezone(timedelta(hours=8))),
		"base_hr": 72.0,
		"variability": 3.2,
		"resp_hz": 0.23,
		"motion": 0.20,
		"delay_s": 0.18,
	},
	{
		"idx": 2,
		"start": datetime(2026, 6, 17, 19, 0, 0, tzinfo=timezone(timedelta(hours=8))),
		"base_hr": 86.0,
		"variability": 4.8,
		"resp_hz": 0.27,
		"motion": 0.28,
		"delay_s": 0.21,
	},
	{
		"idx": 3,
		"start": datetime(2026, 6, 17, 20, 0, 0, tzinfo=timezone(timedelta(hours=8))),
		"base_hr": 101.0,
		"variability": 5.5,
		"resp_hz": 0.31,
		"motion": 0.34,
		"delay_s": 0.24,
	},
]


def read_json(path):
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path, payload):
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=4)


def moving_average_frames(data, window):
	pad = window // 2
	padded = np.pad(data, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
	acc = np.zeros_like(data, dtype=np.float32)
	for offset in range(window):
		acc += padded[offset:offset + data.shape[0]]
	return acc / float(window)


def load_ecg_noise():
	values = []
	with ECG_REFERENCE.open("r", encoding="utf-8") as f:
		next(f)
		for line in f:
			if not line.strip():
				continue
			_, rest = line.split(":", 1)
			values.append(int(rest.split()[0]))
	ecg = np.asarray(values, dtype=np.float32)
	ecg = np.clip(ecg, np.percentile(ecg, 1), np.percentile(ecg, 99))
	window = 201
	kernel = np.ones(window, dtype=np.float32) / window
	baseline = np.convolve(ecg, kernel, mode="same")
	noise = ecg - baseline
	noise -= noise.mean()
	noise_std = noise.std() or 1.0
	return noise / noise_std


def smooth_random_walk(rng, n, scale):
	steps = rng.normal(0.0, scale, size=n)
	walk = np.cumsum(steps)
	walk -= np.linspace(walk[0], walk[-1], n)
	return walk


def attenuate_range_bins(samples, bins, factor=0.18):
	spec = np.fft.fft(samples, axis=-1)
	n = samples.shape[-1]
	for bin_idx in bins:
		spec[..., bin_idx] *= factor
		spec[..., n - bin_idx] *= factor
	return np.fft.ifft(spec, axis=-1).real.astype(np.float32)


def heart_rate_profile(times, base_hr, variability, rng):
	slow = variability * np.sin(2 * np.pi * times / rng.uniform(25.0, 38.0) + rng.uniform(0, 2 * np.pi))
	faster = 1.2 * np.sin(2 * np.pi * times / rng.uniform(8.0, 14.0) + rng.uniform(0, 2 * np.pi))
	drift = smooth_random_walk(rng, len(times), 0.025)
	return np.clip(base_hr + slow + faster + drift, 58.0, 125.0)


def make_beat_schedule(duration, base_hr, variability, rng):
	beats = []
	rr_values = []
	t = 0.55 + rng.normal(0.0, 0.04)
	prev_rr = 60.0 / base_hr
	while t < duration + 1.0:
		slow_hr = variability * np.sin(2 * np.pi * t / rng.uniform(24.0, 36.0) + rng.uniform(0, 2 * np.pi))
		resp_hr = 0.9 * np.sin(2 * np.pi * 0.22 * t + rng.uniform(0, 2 * np.pi))
		target_rr = 60.0 / np.clip(base_hr + slow_hr + resp_hr, 55.0, 130.0)
		jitter = rng.normal(0.0, 0.028 + 0.0025 * variability)
		rr = 0.72 * prev_rr + 0.28 * target_rr + jitter
		rr = float(np.clip(rr, 0.45, 1.25))
		t += rr
		beats.append(t)
		rr_values.append(rr)
		prev_rr = rr
	beats = np.asarray(beats, dtype=float)
	rr_values = np.asarray(rr_values, dtype=float)
	beats = beats[beats < duration + 0.8]
	rr_values = rr_values[:len(beats)]
	return beats, rr_values


def hr_from_beats(times, beat_times, rr_values, base_hr):
	if len(beat_times) < 2:
		return np.full_like(times, base_hr, dtype=float)
	hr_at_beats = 60.0 / np.clip(rr_values, 1e-6, None)
	return np.interp(times, beat_times, hr_at_beats, left=hr_at_beats[0], right=hr_at_beats[-1])


def integrate_phase(times, hr_bpm):
	dt = np.diff(times, prepend=times[0])
	return 2 * np.pi * np.cumsum((hr_bpm / 60.0) * dt)


def interp_profile(times, source_times, values):
	return np.interp(times, source_times, values, left=values[0], right=values[-1])


def mechanical_heartbeat_wave(times, beat_times, rng, base_delay_s):
	times = np.asarray(times, dtype=float)
	wave = np.zeros_like(times)
	for beat_idx, beat_time in enumerate(beat_times):
		delay = base_delay_s + rng.normal(0.0, 0.018)
		amp = 1.0 + 0.13 * np.sin(0.39 * beat_idx + 0.4) + rng.normal(0.0, 0.055)
		rise = np.clip(rng.normal(0.038, 0.007), 0.022, 0.060)
		decay = np.clip(rng.normal(0.205, 0.035), 0.135, 0.310)
		rebound_delay = np.clip(rng.normal(0.36, 0.035), 0.26, 0.48)
		rebound_width = np.clip(rng.normal(0.085, 0.018), 0.050, 0.130)
		dt = times - (beat_time + delay)
		pos = dt >= 0
		pulse = np.zeros_like(times)
		pulse[pos] = np.exp(-dt[pos] / decay) - np.exp(-dt[pos] / rise)
		if np.max(pulse) > 0:
			pulse /= np.max(pulse)
		rebound = -0.23 * np.exp(-0.5 * ((times - (beat_time + delay + rebound_delay)) / rebound_width) ** 2)
		wave += amp * pulse + amp * rebound
	wave -= np.mean(wave)
	std = np.std(wave)
	if std > 0:
		wave /= std
	return wave


def synthetic_ecg(times, beat_times, rng, noise_reference):
	ecg = np.zeros_like(times)
	for beat_idx, beat_time in enumerate(beat_times):
		amp = 1.0 + 0.06 * np.sin(0.47 * beat_idx) + rng.normal(0.0, 0.025)
		width = 1.0 + rng.normal(0.0, 0.03)
		ecg += 170.0 * amp * np.exp(-0.5 * ((times - (beat_time - 0.17)) / (0.045 * width)) ** 2)
		ecg += -420.0 * amp * np.exp(-0.5 * ((times - (beat_time - 0.028)) / (0.012 * width)) ** 2)
		ecg += 2600.0 * amp * np.exp(-0.5 * ((times - beat_time) / (0.014 * width)) ** 2)
		ecg += -720.0 * amp * np.exp(-0.5 * ((times - (beat_time + 0.032)) / (0.017 * width)) ** 2)
		ecg += 430.0 * amp * np.exp(-0.5 * ((times - (beat_time + 0.27)) / (0.085 * width)) ** 2)

	baseline = 110.0 * np.sin(2 * np.pi * 0.18 * times + rng.uniform(0, 2 * np.pi))
	baseline += 42.0 * np.sin(2 * np.pi * 0.045 * times + rng.uniform(0, 2 * np.pi))
	start = rng.integers(0, max(1, len(noise_reference) - len(times) - 1))
	ref_noise = noise_reference[start:start + len(times)]
	if len(ref_noise) < len(times):
		ref_noise = np.resize(noise_reference, len(times))
	noise = 30.0 * rng.normal(size=len(times)) + 65.0 * ref_noise
	for center in rng.uniform(4.0, DURATION_S - 4.0, size=3):
		noise += rng.normal(0, 80) * np.exp(-0.5 * ((times - center) / rng.uniform(0.05, 0.16)) ** 2)
	ecg = ecg + baseline + noise
	ecg -= np.median(ecg)
	return np.clip(ecg, -4200, 4200).astype(int)


def write_ecg(path, start_epoch, duration, base_hr, beat_times, rr_values, rng, noise_reference):
	n = int(round(duration * ECG_FS))
	times = np.arange(n) / ECG_FS
	hr = hr_from_beats(times, beat_times, rr_values, base_hr)
	ecg = synthetic_ecg(times, beat_times, rng, noise_reference)

	with path.open("w", encoding="utf-8") as f:
		f.write("timestamp: ADC HeartRate4sAverage HeartRate30sAverage\n")
		for idx, (relative_t, adc, hr_value) in enumerate(zip(times, ecg, hr)):
			group_t = np.floor(relative_t / 0.032) * 0.032
			timestamp = start_epoch + group_t
			left4 = max(0, idx - int(4 * ECG_FS))
			left30 = max(0, idx - int(30 * ECG_FS))
			hr4 = int(round(float(np.mean(hr[left4:idx + 1]))))
			hr30 = int(round(float(np.mean(hr[left30:idx + 1]))))
			f.write(f"{timestamp:.3f}: {int(adc):6d} {hr4:3d} {hr30:3d}\n")
	return times, hr


def synth_radar(template, config, session, beat_times, rr_values, rng):
	shape = config["device_config"]["fmcw_single_shape"]
	frame_rt = float(shape["frame_repetition_time_s"])
	chirp_rt = float(shape["chirp_repetition_time_s"])
	n_frames = int(round(DURATION_S / frame_rt))
	rx_count, chirps, samples = template.shape[1:]
	frame_times = np.arange(n_frames) * frame_rt
	chirp_offsets = np.arange(chirps) * chirp_rt
	chirp_times = frame_times[:, None] + chirp_offsets[None, :]

	template_float = template.astype(np.float32)
	smooth = moving_average_frames(template_float, 11)
	static = attenuate_range_bins(np.median(smooth, axis=0), (4, 5), factor=0.16)
	residual = template_float - smooth
	residual = np.clip(residual, np.percentile(residual, 0.5), np.percentile(residual, 99.5))
	residual = attenuate_range_bins(residual, (4, 5), factor=0.28)

	flat_times = chirp_times.reshape(-1)
	mech_flat = mechanical_heartbeat_wave(flat_times, beat_times, rng, session["delay_s"])
	mech = mech_flat.reshape(n_frames, chirps)
	hr_frame = hr_from_beats(frame_times, beat_times, rr_values, session["base_hr"])

	sample_axis = np.arange(samples, dtype=np.float32)
	target_bin = 4 + rng.normal(0.0, 0.08)
	neighbor_bin = 5 + rng.normal(0.0, 0.06)
	rx_gain = np.asarray([1.0, 0.93, 0.86], dtype=np.float32)[:rx_count]
	rx_phase = np.asarray([0.0, 0.23, -0.19], dtype=np.float32)[:rx_count]
	noise_scale = 0.34 + 0.06 * session["motion"]
	output = np.empty((n_frames, rx_count, chirps, samples), dtype=np.uint16)

	source_positions = np.linspace(0, template.shape[0] - 1, n_frames)
	source_positions += 1.8 * np.sin(2 * np.pi * frame_times / 17.0 + rng.uniform(0, 2 * np.pi))
	source_positions = np.clip(np.round(source_positions).astype(int), 0, template.shape[0] - 1)
	wander = smooth_random_walk(rng, n_frames * chirps, 0.004).reshape(n_frames, chirps)
	amp_mod_flat = 1.0 + 0.055 * mechanical_heartbeat_wave(flat_times, beat_times, rng, session["delay_s"] + 0.025)
	amp_mod = np.clip(amp_mod_flat.reshape(n_frames, chirps), 0.82, 1.22)

	for f_idx in range(n_frames):
		src = source_positions[f_idx]
		base = static + noise_scale * residual[src]
		base += 0.16 * residual[rng.integers(0, template.shape[0])]
		resp = 0.42 * np.sin(2 * np.pi * session["resp_hz"] * chirp_times[f_idx] + 0.4 * session["idx"])
		heart = 0.46 * mech[f_idx]
		motion = session["motion"] * 0.06 * np.sin(2 * np.pi * 0.065 * chirp_times[f_idx] + 1.7)
		phase_mod = resp + heart + motion + wander[f_idx]
		for rx in range(rx_count):
			for c_idx in range(chirps):
				amp = 150.0 * rx_gain[rx] * amp_mod[f_idx, c_idx] * (1.0 + 0.025 * rng.normal())
				amp *= 1.0 + rng.normal(0.0, 0.012)
				phase_c = phase_mod[c_idx] + rx_phase[rx] + rng.normal(0.0, 0.01)
				target = amp * np.cos(2 * np.pi * target_bin * sample_axis / samples + phase_c)
				target += 0.62 * amp * np.cos(2 * np.pi * neighbor_bin * sample_axis / samples + phase_c + 0.48)
				fine_noise = rng.normal(0.0, 7.5 + 10.0 * session["motion"], size=samples)
				values = base[rx, c_idx] + target + fine_noise
				output[f_idx, rx, c_idx] = np.clip(np.rint(values), 0, 4095).astype(np.uint16)
	return output, hr_frame


def copy_template_files(radar_dir):
	shutil.copy2(TEMPLATE_RADAR_DIR / "config.json", radar_dir / "config.json")
	shutil.copy2(TEMPLATE_RADAR_DIR / "format.version", radar_dir / "format.version")


def generate_session(template, config, template_meta, root_meta, noise_reference, session):
	idx = session["idx"]
	rng = np.random.default_rng(2026061700 + idx)
	start_dt = session["start"]
	record_name = f"BGT60TR13C_record_{start_dt.strftime('%Y%m%d-%H%M%S')}_realistic_sim{idx}"
	record_dir = OUT_ROOT / record_name
	radar_dir = record_dir / "RadarIfxAvian_00"
	radar_dir.mkdir(parents=True, exist_ok=True)

	beat_times, rr_values = make_beat_schedule(DURATION_S, session["base_hr"], session["variability"], rng)
	radar, _ = synth_radar(template, config, session, beat_times, rr_values, rng)
	np.save(radar_dir / "radar.npy", radar)
	copy_template_files(radar_dir)

	radar_meta = dict(template_meta)
	radar_meta["description"] = "Realistic template-based simulated BGT60TR13C FMCW Radar Sensor"
	radar_meta["uuid"] = f"realistic-sim-{idx}"
	radar_meta["source_template"] = str(TEMPLATE_RADAR_DIR.relative_to(ROOT))
	radar_meta["heartbeat_model"] = "asymmetric mechanical pulse with beat-to-beat HRV"
	write_json(radar_dir / "meta.json", radar_meta)

	record_meta = dict(root_meta)
	record_meta["date_captured"] = start_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
	record_meta["hostname"] = "template-simulated"
	record_meta["username"] = "template-simulated"
	record_meta["source_template"] = str(TEMPLATE_RECORD_DIR.relative_to(ROOT))
	write_json(record_dir / "meta.json", record_meta)

	ecg_path = OUT_ROOT / f"ECGLog-realistic-sim-{idx}-{start_dt.strftime('%Y-%m-%d-%H-%M-%S')}.txt"
	write_ecg(
		ecg_path,
		start_dt.timestamp(),
		DURATION_S,
		session["base_hr"],
		beat_times,
		rr_values,
		rng,
		noise_reference,
	)
	return radar_dir / "radar.npy", ecg_path


def main():
	OUT_ROOT.mkdir(parents=True, exist_ok=True)
	template = np.load(TEMPLATE_RADAR_DIR / "radar.npy", mmap_mode="r")
	config = read_json(TEMPLATE_RADAR_DIR / "config.json")
	template_meta = read_json(TEMPLATE_RADAR_DIR / "meta.json")
	root_meta = read_json(TEMPLATE_RECORD_DIR / "meta.json")
	noise_reference = load_ecg_noise()

	print("Generated template-based simulated data:")
	for session in SESSIONS:
		radar_path, ecg_path = generate_session(template, config, template_meta, root_meta, noise_reference, session)
		print(f"  radar: {radar_path}")
		print(f"  ecg:   {ecg_path}")


if __name__ == "__main__":
	main()
