import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
	from scipy.signal import butter, filtfilt
except Exception:
	butter = None
	filtfilt = None


def load_json(path):
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def bandpass(x, fs, low, high, order=4):
	if butter is None or filtfilt is None:
		# fallback: simple detrend (no real bandpass)
		return x - np.mean(x)
	ny = 0.5 * fs
	b, a = butter(order, [low / ny, high / ny], btype='band')
	return filtfilt(b, a, x)


def load_record(record_dir):
	record_dir = Path(record_dir)
	meta = load_json(record_dir / 'meta.json') if (record_dir / 'meta.json').exists() else {}
	config = load_json(record_dir / 'config.json') if (record_dir / 'config.json').exists() else {}
	radar_path = record_dir / 'radar.npy'
	if not radar_path.exists():
		raise FileNotFoundError(f'radar.npy not found in {record_dir}')
	arr = np.load(radar_path, mmap_mode='r')
	return meta, config, arr


def compute_range_fft(samples):
	# samples: (num_samples,) or (..., num_samples)
	win = np.hanning(samples.shape[-1])
	return np.fft.fft(samples * win, axis=-1)


def compute_range_doppler(frame_data):
	# frame_data: (rx, num_chirps, num_samples)
	# for simplicity pick rx=0
	rx0 = frame_data[0]
	# range FFT per chirp -> (num_chirps, num_range_bins)
	rng = np.fft.fft(rx0 * np.hanning(rx0.shape[-1]), axis=-1)
	# doppler across chirps per range bin
	dop = np.fft.fftshift(np.fft.fft(rng, axis=0), axes=0)
	# return magnitude (doppler, range)
	return np.abs(dop), rng


def compute_range_angle(frame_data):
	# frame_data: (rx, num_chirps, num_samples)
	# integrate across chirps and take FFT across rx for each range bin
	# average across chirps
	avg_chirp = np.mean(frame_data, axis=1)  # (rx, num_samples)
	# range FFT per rx -> (rx, num_range)
	rng = np.fft.fft(avg_chirp * np.hanning(avg_chirp.shape[-1]), axis=-1)
	# angle FFT across rx -> (angle_bins, num_range)
	angle_map = np.fft.fftshift(np.fft.fft(rng, n=64, axis=0), axes=0)
	return np.abs(angle_map)


def main(record_dir=None, out_dir=None):
	def choose_record_dir_interactive():
		# try GUI folder dialog first
		try:
			import tkinter as tk
			from tkinter import filedialog
			root = tk.Tk()
			root.withdraw()
			p = filedialog.askdirectory(title='Select Radar Record Folder')
			root.destroy()
			if p:
				return Path(p)
		except Exception:
			pass
		# fallback to CLI input
		s = input('Enter record folder path (or press Enter to cancel): ').strip()
		if s:
			return Path(s)
		raise ValueError('No record folder provided')

	if record_dir is None:
		record_dir = choose_record_dir_interactive()
	record_dir = Path(record_dir)
	out_dir = Path(out_dir or (Path.cwd() / 'bgtr13c_outputs'))
	out_dir.mkdir(parents=True, exist_ok=True)

	meta, config, arr = load_record(record_dir)
	print('meta:', meta)
	print('config sample keys:', list(config.keys()))
	print('raw radar shape:', arr.shape, 'dtype:', arr.dtype)

	# Expecting shape (frames, rx, chirps, samples)
	frames, rx_count, chirps, samples = arr.shape
	frame_rt = config.get('device_config', {}).get('fmcw_single_shape', {}).get('frame_repetition_time_s', None)
	if frame_rt is None:
		frame_rt = config.get('device_config', {}).get('fmcw_single_shape', {}).get('chirp_repetition_time_s', 0.0005911249900236726) * chirps
	trim_secs = 10.0
	cut_frames = int(round(trim_secs / frame_rt))
	if cut_frames * 2 >= frames:
		raise ValueError(f'Not enough frames to trim {trim_secs}s from head and tail: {frames} frames total')
	print(f'trim first/last {trim_secs}s => {cut_frames} frames each')
	arr = arr[cut_frames:frames-cut_frames]
	frames = arr.shape[0]
	print('trimmed radar shape:', arr.shape)

	# 1. 原始波形图: first frame, first rx, first chirp samples
	raw = arr[0, 0, 0, :].astype(float)
	# save raw samples for debugging
	np.savetxt(out_dir / 'raw_samples_frame0_rx0_chirp0.txt', raw, fmt='%f')
	plt.figure(figsize=(8,3))
	plt.plot(raw)
	plt.title('Raw ADC waveform (frame0 rx0 chirp0)')
	plt.xlabel('sample index')
	plt.ylabel('ADC')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'raw_waveform.png')

	# 2. 计算 range-fft 并选择目标 range bin
	# compute range FFT for all frames, rx, chirps
	# to save memory, compute magnitude averaged across chirps and rx
	rng_mag = np.zeros((frames, samples))
	for f in range(frames):
		# pick rx0 for speed
		data = arr[f, 0]  # (chirps, samples)
		rng = np.fft.fft(data * np.hanning(samples), axis=-1)
		rng_mag[f] = np.mean(np.abs(rng), axis=0)

	avg_mag = np.mean(rng_mag, axis=0)
	# Avoid selecting DC/nyquist bins (edges). Prefer bins with time-varying phase.
	exclude_bins = set([0, 1, samples-1, samples-2])
	# compute phase std for each range bin across all chirps (frames*chirps points)
	phase_stds = np.zeros(samples)
	for b in range(samples):
		vals_b = []
		for f in range(frames):
			data = arr[f, 0]  # (chirps, samples)
			rng = np.fft.fft(data * np.hanning(samples), axis=-1)
			vals_b.extend(rng[:, b].tolist())
		vals_b = np.array(vals_b)
		ph = np.unwrap(np.angle(vals_b))
		phase_stds[b] = float(ph.std())
	# save phase std for debugging
	np.savetxt(out_dir / 'range_bin_phase_std.txt', phase_stds, fmt='%f')
	# choose bin with largest phase std excluding edges
	cand = [i for i in range(samples) if i not in exclude_bins]
	if len(cand) == 0:
		target_range_bin = int(np.argmax(avg_mag))
	else:
		target_range_bin = int(cand[np.argmax(phase_stds[cand])])
	print('selected range bin (by phase std):', target_range_bin)
	# also save avg_mag plot
	# save avg range magnitude for debugging
	np.savetxt(out_dir / 'avg_range_mag.txt', avg_mag, fmt='%f')
	plt.figure(figsize=(8,3))
	plt.plot(avg_mag)
	plt.title('Average Range Magnitude (across frames)')
	plt.xlabel('range bin')
	plt.ylabel('magnitude')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'avg_range_magnitude.png')

	# Build time series by flattening frames and chirps in time order
	chirp_rt = config.get('device_config', {}).get('fmcw_single_shape', {}).get('chirp_repetition_time_s', None)
	if chirp_rt is None:
		chirp_rt = 0.0005911249900236726
	fs = 1.0 / chirp_rt
	total_chirps = frames * chirps
	time = np.arange(total_chirps) * chirp_rt

	# extract complex value at target_range_bin for rx0, per chirp
	vals = np.zeros(total_chirps, dtype=complex)
	idx = 0
	for f in range(frames):
		# data: (rx, chirps, samples)
		for c in range(chirps):
			s = arr[f, 0, c, :].astype(float)
			rng = np.fft.fft(s * np.hanning(samples))
			vals[idx] = rng[target_range_bin]
			idx += 1

	# Also compute a frame-based complex series (average across chirps per frame)
	frame_vals = np.zeros(frames, dtype=complex)
	for f in range(frames):
		data = arr[f, 0]  # (chirps, samples)
		rng = np.fft.fft(data * np.hanning(samples), axis=-1)  # (chirps, range)
		mean_rng = np.mean(rng, axis=0)
		frame_vals[f] = mean_rng[target_range_bin]
	frame_time = np.arange(frames) * config.get('device_config', {}).get('fmcw_single_shape', {}).get('frame_repetition_time_s', 0.07726884633302689)
	np.savetxt(out_dir / 'frame_series_vals_real_imag.csv',
			   np.vstack([frame_time, frame_vals.real, frame_vals.imag, np.abs(frame_vals), np.angle(frame_vals)]).T,
			   delimiter=',', header='time,real,imag,mag,phase', comments='')
	# save a quick plot
	plt.figure(figsize=(8,3))
	plt.plot(frame_time, np.angle(frame_vals))
	plt.title('Frame-based phase (per-frame average across chirps)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'frame_phase.png')

	# Save complex values and basic stats for debugging
	mags = np.abs(vals)
	phases = np.angle(vals)
	unw_phases = np.unwrap(phases)
	np.savetxt(out_dir / 'vals_real_imag_mag_phase.csv',
			   np.vstack([vals.real, vals.imag, mags, phases, unw_phases]).T,
			   delimiter=',', header='real,imag,mag,phase,unwrapped_phase', comments='')

	# Prepare phase series and intermediate steps
	phase_unwrapped = unw_phases
	phase_detrended = phase_unwrapped - np.mean(phase_unwrapped)
	# bandpass filter around typical heart-rate band (0.8-3.5 Hz)
	filtered = bandpass(phase_detrended, fs, 0.8, 3.5)

	# Save time series data for each step
	np.savetxt(out_dir / 'time_series_phase_unwrapped.txt', np.column_stack([time, phase_unwrapped]),
			   delimiter=',', header='time,phase_unwrapped', comments='')
	np.savetxt(out_dir / 'time_series_phase_detrended.txt', np.column_stack([time, phase_detrended]),
			   delimiter=',', header='time,phase_detrended', comments='')
	np.savetxt(out_dir / 'time_series_phase_filtered.txt', np.column_stack([time, filtered]),
			   delimiter=',', header='time,phase_filtered', comments='')

	# Plot intermediate phase steps
	plt.figure(figsize=(8,3))
	plt.plot(time, phases)
	plt.title('Phase (raw)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'phase_raw.png')

	plt.figure(figsize=(8,3))
	plt.plot(time, phase_unwrapped)
	plt.title('Phase (unwrapped)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'phase_unwrapped.png')

	plt.figure(figsize=(8,3))
	plt.plot(time, phase_detrended)
	plt.title('Phase (detrended)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'phase_detrended.png')

	plt.figure(figsize=(8,3))
	plt.plot(time, filtered)
	plt.title('Phase (filtered)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'phase_filtered.png')

	plt.figure(figsize=(8,3))
	plt.plot(time, filtered)
	plt.title('Filtered heart-rate-like waveform (phase)')
	plt.xlabel('time (s)')
	plt.ylabel('filtered phase')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'heart_rate_waveform.png')

	# 3. velocity-range (range-Doppler) for frame 0, rx0
	frame0 = arr[0]
	dop_mag, rng = compute_range_doppler(frame0)
	# dop_mag shape: (doppler_bins, range_bins)
	plt.figure(figsize=(6,5))
	plt.imshow(20*np.log10(dop_mag + 1e-6), aspect='auto', origin='lower')
	plt.title('Velocity-Range (dB) frame0 rx0')
	plt.xlabel('range bin')
	plt.ylabel('doppler bin')
	plt.colorbar(label='dB')
	plt.tight_layout()
	plt.savefig(out_dir / 'velocity_range.png')

	# 4. range-angle map for frame0
	angle_map = compute_range_angle(frame0)
	# angle_map shape: (angle_bins, range_bins)
	plt.figure(figsize=(6,5))
	plt.imshow(20*np.log10(angle_map + 1e-6), aspect='auto', origin='lower')
	plt.title('Range-Angle Map (dB) frame0')
	plt.xlabel('range bin')
	plt.ylabel('angle bin')
	plt.colorbar(label='dB')
	plt.tight_layout()
	plt.savefig(out_dir / 'range_angle.png')

	print('Saved outputs to', out_dir)


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser(description='BGT60TR13C record processing demo')
	p.add_argument('--record', help='record folder', default=None)
	p.add_argument('--out', help='output folder', default=None)
	args = p.parse_args()
	main(record_dir=args.record, out_dir=args.out)
