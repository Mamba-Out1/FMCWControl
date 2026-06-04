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


def compute_range_doppler_cube(frame_data, chirp_rt, half_range=True):
	# frame_data: (rx, num_chirps, num_samples)
	rx, num_chirps, num_samples = frame_data.shape
	window_range = np.hanning(num_samples)
	window_doppler = np.hanning(num_chirps)
	# Range FFT for each rx and each chirp
	rng = np.fft.fft(frame_data * window_range[np.newaxis, np.newaxis, :], axis=2)
	if half_range:
		rng = rng[..., :num_samples//2]
	# Doppler FFT across chirps for each rx and range bin
	dop = np.fft.fftshift(np.fft.fft(rng * window_doppler[np.newaxis, :, np.newaxis], axis=1), axes=1)
	# Return magnitude cube: (rx, doppler_bins, range_bins)
	return np.abs(dop), rng


def compute_range_doppler(frame_data, chirp_rt, half_range=True):
	# legacy wrapper for single RX return
	dop_cube, rng = compute_range_doppler_cube(frame_data, chirp_rt, half_range=half_range)
	return dop_cube[0], rng[0]


def compute_range_angle(frame_data):
	# frame_data: (rx, num_chirps, num_samples)
	# Mild DC removal: only per-sample mean (avoid over-removal)
	frame_dc = frame_data - np.mean(frame_data, axis=2, keepdims=True)  # per-sample mean only
	# average across chirps (integrate incoherently)
	avg_chirp = np.mean(frame_dc, axis=1)  # (rx, num_samples)
	# Normalize each rx by its RMS to equalize multi-antenna gain
	rms_per_rx = np.sqrt(np.mean(avg_chirp**2, axis=1, keepdims=True)) + 1e-6
	avg_chirp_norm = avg_chirp / rms_per_rx
	# range FFT per rx -> (rx, num_range)
	rng = np.fft.fft(avg_chirp_norm * np.hanning(avg_chirp_norm.shape[-1]), axis=-1)
	# angle FFT across rx -> (angle_bins, num_range)
	# Use number of rx antennas for FFT size, pad to 64 for display
	num_rx = rng.shape[0]
	angle_map = np.fft.fftshift(np.fft.fft(rng, n=64, axis=0), axes=0)
	return np.abs(angle_map)


def assess_image_quality(map_data, name='', out_dir=None):
	"""Assess quality of range-doppler or range-angle map."""
	mag_db = 20*np.log10(map_data + 1e-6)
	peak = float(np.max(mag_db))
	mean_bg = float(np.percentile(mag_db, 50))  # median as background
	noise = float(np.percentile(mag_db, 25))
	snr = peak - noise
	# Find number of bins above noise+10dB
	thresh = noise + 10
	peak_bins = np.sum(mag_db > thresh)
	concentration = peak_bins / map_data.size * 100
	
	print(f'{name} Quality Metrics:')
	print(f'  Peak: {peak:.1f} dB')
	print(f'  Noise (25th percentile): {noise:.1f} dB')
	print(f'  SNR (peak - noise): {snr:.1f} dB')
	print(f'  Energy concentration: {concentration:.2f}% of bins above noise+10dB')
	print(f'  Quality: {"POOR" if snr < 10 else "FAIR" if snr < 20 else "GOOD"}')
	
	if out_dir:
		# Save histogram
		plt.figure(figsize=(8,3))
		plt.hist(mag_db.flatten(), bins=50, color='steelblue', edgecolor='black')
		plt.axvline(peak, color='r', linestyle='--', label=f'Peak={peak:.1f}dB')
		plt.axvline(noise, color='orange', linestyle='--', label=f'Noise={noise:.1f}dB')
		plt.xlabel('Magnitude (dB)')
		plt.ylabel('Count')
		plt.title(f'{name} Magnitude Histogram (SNR={snr:.1f}dB)')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(out_dir / f'{name}_histogram.png')
		plt.close()
	
	return {'peak': peak, 'noise': noise, 'snr': snr, 'concentration': concentration}


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
	# compute range FFT for all frames, rx, chirps and keep half-spectrum
	half_bins = samples // 2
	range_bandwidth = None
	start_freq = config.get('device_config', {}).get('fmcw_single_shape', {}).get('start_frequency_Hz', None)
	end_freq = config.get('device_config', {}).get('fmcw_single_shape', {}).get('end_frequency_Hz', None)
	if start_freq is not None and end_freq is not None:
		range_bandwidth = end_freq - start_freq
	if range_bandwidth is None or range_bandwidth <= 0:
		range_bandwidth = 5.5e9
	range_resolution = 299792458.0 / (2.0 * range_bandwidth)
	print(f'Range resolution: {range_resolution:.4f} m/bin')
	
	rng_mag = np.zeros((frames, rx_count, half_bins))
	window = np.hanning(samples)
	for f in range(frames):
		data = arr[f]  # (rx, chirps, samples)
		rng = np.fft.fft(data * window[np.newaxis, None, :], axis=-1)[..., :half_bins]
		rng_mag[f] = np.mean(np.abs(rng), axis=1)
	
	avg_mag = np.mean(rng_mag, axis=(0,1))
	# save avg range magnitude for debugging
	np.savetxt(out_dir / 'avg_range_mag.txt', avg_mag, fmt='%f')
	
	# choose target bin based on peak magnitude in valid range
	search_start = 2
	search_end = half_bins
	# if we know target is near 30cm, narrow search window
	target_range_m = 0.30
	target_bin_est = int(round(target_range_m / range_resolution))
	search_radius = max(3, int(round(0.20 / range_resolution)))
	search_start = max(search_start, target_bin_est - search_radius)
	search_end = min(search_end, target_bin_est + search_radius + 1)
	if search_start >= search_end:
		search_start = 2
		search_end = half_bins
	target_range_bin = search_start + int(np.argmax(avg_mag[search_start:search_end]))
	print(f'selected target range bin: {target_range_bin} (search range {search_start}-{search_end-1})')
	
	# use distance axis for plots
	range_axis = np.arange(half_bins) * range_resolution
	print(f'Estimated target distance: {target_range_bin * range_resolution:.3f} m')
	
	# === NEW: 1. Single-frame Range FFT spectrum ===
	frame0_data = arr[0, 0]  # (chirps, samples)
	frame0_rng = np.fft.fft(frame0_data * np.hanning(samples), axis=-1)[:, :half_bins]
	frame0_rng_mag = np.mean(np.abs(frame0_rng), axis=0)  # average across chirps
	
	plt.figure(figsize=(10, 4))
	plt.plot(range_axis, frame0_rng_mag, 'b-', label='Range FFT magnitude')
	plt.axvline(target_range_bin * range_resolution, color='r', linestyle='--', linewidth=2, label=f'Target bin {target_range_bin}')
	plt.xlabel('Distance (m)')
	plt.ylabel('Magnitude')
	plt.title('Range FFT Spectrum (frame 0, rx 0, avg across chirps)')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_dir / '1_range_fft_spectrum.png')
	plt.close()
	print(f'Saved Range FFT spectrum with target bin {target_range_bin} marked')
	
	# === Average range magnitude plot with target bin marked ===
	plt.figure(figsize=(10, 4))
	plt.plot(range_axis, avg_mag, 'b-', label='Avg range magnitude')
	plt.axvline(target_range_bin * range_resolution, color='r', linestyle='--', linewidth=2, label=f'Selected bin {target_range_bin}')
	plt.xlabel('Distance (m)')
	plt.ylabel('magnitude')
	plt.title('Average Range Magnitude (across all frames) - Target Bin Selection')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / '2_avg_range_magnitude_marked.png')
	plt.close()
	
	# Get chirp and frametime for 30s window
	frame_rt_val = config.get('device_config', {}).get('fmcw_single_shape', {}).get('frame_repetition_time_s', 0.16129031777381897)
	secs_30 = 30.0
	num_frames_30s = int(round(secs_30 / frame_rt_val))
	num_frames_30s = min(num_frames_30s, frames)
	print(f'Extracting {num_frames_30s} frames (~{num_frames_30s * frame_rt_val:.1f}s) for 30s phase plot')


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

	# Also compute a frame-based complex series (average across chirps per frame) for debug only
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
	# save a debug plot for frame-based phase (not heart rate)
	plt.figure(figsize=(8,3))
	plt.plot(frame_time, np.angle(frame_vals))
	plt.title('Frame-based phase (per-frame average across chirps)')
	plt.xlabel('time (s)')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(out_dir / 'frame_phase.png')
	plt.close()

	# === NEW: 3. 30-second chirp-level phase curve for target range bin ===
	num_chirps_30s = int(round(secs_30 / chirp_rt))
	num_chirps_30s = min(num_chirps_30s, total_chirps)
	chirp_time_30s = np.arange(num_chirps_30s) * chirp_rt
	phase_30s = np.unwrap(np.angle(vals[:num_chirps_30s]))
	phase_30s_detrended = phase_30s - np.polyfit(chirp_time_30s, phase_30s, 1)[0] * chirp_time_30s - np.polyfit(chirp_time_30s, phase_30s, 1)[1]

	plt.figure(figsize=(12, 5))
	plt.plot(chirp_time_30s, phase_30s_detrended, 'b-', linewidth=1.5, label='Detrended chirp-level phase')
	plt.xlabel('Time (s)')
	plt.ylabel('Phase (rad)')
	plt.title(f'Chirp-level phase curve - Range Bin {target_range_bin} (30s)')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / '3_phase_30s_target_bin.png')
	plt.close()
	print(f'Saved 30s chirp-level phase curve for bin {target_range_bin}')

	# Also save raw phase data for monitoring
	np.savetxt(out_dir / 'phase_30s_target_bin.csv',
			   np.vstack([chirp_time_30s, phase_30s, phase_30s_detrended]).T,
			   delimiter=',', header='time,phase_raw,phase_detrended', comments='')

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

	# 3. velocity-range (range-Doppler) using full 2D FFT in each frame
	num_frames_for_map = min(10, frames)
	doppler_axis = np.fft.fftshift(np.fft.fftfreq(chirps, d=chirp_rt))
	rd_cube_all = []
	for f in range(num_frames_for_map):
		frame_data = arr[f]
		rd_cube, _ = compute_range_doppler_cube(frame_data, chirp_rt, half_range=True)
		rd_cube_all.append(rd_cube)
	rd_avg = np.mean(np.stack(rd_cube_all, axis=0), axis=0)  # (rx, doppler, range)

	# Save as separate RX images and as RGB composite
	for rx_idx in range(rx_count):
		mag = rd_avg[rx_idx]
		plt.figure(figsize=(8,5))
		plt.imshow(20*np.log10(mag + 1e-6), aspect='auto', origin='lower',
				extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]])
		plt.title(f'Range-Doppler (dB) RX{rx_idx}')
		plt.xlabel('Distance (m)')
		plt.ylabel('Doppler Frequency (Hz)')
		plt.colorbar(label='dB')
		plt.tight_layout()
		plt.savefig(out_dir / f'velocity_range_rx{rx_idx}.png')
		plt.close()

	# RGB composite from three RX channels if available
	if rx_count >= 3:
		rgb = np.stack([rd_avg[0], rd_avg[1], rd_avg[2]], axis=-1)
		rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
		plt.figure(figsize=(8,5))
		plt.imshow(rgb, aspect='auto', origin='lower',
				extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]])
		plt.title('Range-Doppler RGB composite (RX0=R,RX1=G,RX2=B)')
		plt.xlabel('Distance (m)')
		plt.ylabel('Doppler Frequency (Hz)')
		plt.tight_layout()
		plt.savefig(out_dir / 'velocity_range_rgb.png')
		plt.close()

	# Use RX0 map for quality assessment
	dop_qual = assess_image_quality(rd_avg[0], 'Velocity-Range', out_dir)
	print('Velocity-Range quality assessed on RX0 channel')

	# 4. range-angle map - keep existing angle processing as an auxiliary view
	angle_map_all = []
	for f in range(num_frames_for_map):
		frame_data = arr[f]
		angle_map = compute_range_angle(frame_data)
		angle_map_all.append(angle_map)
	angle_map = np.mean(angle_map_all, axis=0)

	# Assess quality
	angle_qual = assess_image_quality(angle_map, 'Range-Angle', out_dir)

	plt.figure(figsize=(6,5))
	plt.imshow(20*np.log10(angle_map + 1e-6), aspect='auto', origin='lower', vmin=-20, vmax=100)
	plt.title(f'Range-Angle Map (dB) avg({num_frames_for_map} frames)\nSNR={angle_qual["snr"]:.1f}dB, Conc={angle_qual["concentration"]:.1f}%')
	plt.xlabel('range bin')
	plt.ylabel('angle bin')
	plt.colorbar(label='dB')
	plt.tight_layout()
	plt.savefig(out_dir / 'range_angle.png')
	plt.close()

	print('\n' + '='*60)
	print('QUALITY ASSESSMENT & RECOMMENDATIONS')
	print('='*60)
	avg_snr = (dop_qual.get('snr', 0) + angle_qual.get('snr', 0)) / 2
	
	print(f'\nVelocity-Range: SNR={dop_qual.get("snr", 0):.1f}dB')
	if dop_qual.get('snr', 0) >= 20:
		print('  ✓ EXCELLENT: Signal quality is very good!')
		print('  → Ready for heart rate detection on this range bin')
	
	print(f'\nRange-Angle: SNR={angle_qual.get("snr", 0):.1f}dB')
	if angle_qual.get('snr', 0) < 5:
		print('  ✗ POOR: Multi-antenna signal weak or misaligned')
		print('  Likely causes:')
		print('    1. Multi-antenna phase misalignment (hardware/calibration issue)')
		print('    2. Antenna separation too small for angle resolution at this distance')
		print('    3. Target may be at broadside (perpendicular to antenna array)')
		print('  Recommendations:')
		print('    • Check antenna PCB design and calibration')
		print('    • Try different measurement angles (not perpendicular to array)')
		print('    • Verify all RX antennas are enabled in config')
		print('    • For now, rely on Velocity-Range which shows good SNR')
	
	print('\n' + '='*60)
	print('MEASUREMENT DISTANCE ASSESSMENT')
	print('='*60)
	if dop_qual.get('snr', 0) >= 20:
		print('✓ 20cm measurement distance is ADEQUATE')
		print('  → No need to increase distance (V-R SNR is excellent)')
	else:
		print('⚠  Consider increasing to 30-50cm if Velocity-Range SNR improves further')
	print('='*60 + '\n')


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser(description='BGT60TR13C record processing demo')
	p.add_argument('--record', help='record folder', default=None)
	p.add_argument('--out', help='output folder', default=None)
	args = p.parse_args()
	main(record_dir=args.record, out_dir=args.out)
