% 全设置，无分步波形绘�?

clear;
close all;
clc;
rng(42);

%% Parameters
fs_raw = 2000;
max_duration_minutes = 45;
downsample_factor = 16;
fs_ds = fs_raw / downsample_factor;

heart_band_hz = [0.7, 2.5];
valid_hr_bpm = [40, 180];

win_len = 256;
hop_len = 64;
model_file_student = fullfile(pwd, 'heart_resnet_model_kd_student.mat');
model_file_teacher = fullfile(pwd, 'heart_resnet_model_kd_teacher.mat');

% ------- Cross-modal KD settings ---------------
enable_kd = true;
% KD strategy: feature-only distillation
kd_feature_weight = 0.10;  % 0.05 ~ 0.15 recommended
kd_response_weight = 0.00;
% Use a higher layer for feature KD to reduce modality mismatch
kd_feature_layer_unet = 'dec2_relu2';
kd_feature_layer_resnet = 'relu3';

% Teacher quality gate (set NaN to skip a check)
teacher_val_loss_max = NaN;     % disable strict val-loss threshold (scale varies)
teacher_test_corr_min = 0.30;    % disable test correlation check (use NaN to skip)

% Reduce modality gap in feature KD
kd_use_global_pool = true;      % global-average pool features before MSE

% Robust alignment settings
align_num_segments = 3;
align_segment_sec = 20;
align_clip_k = 3.0;

% Outlier suppression (impulsive artifacts)
enable_radar_despike = true;
radar_despike_win_sec = 0.6;
radar_despike_k = 6;

% ------- 设置 --------------------------------

% enable_template_init = true:高斯模板初始化，模板匹配   enable_template_init = false:基线无模板匹�?
% model_arch = 'unet'; u-net架构  model_arch = 'resnet'; resnet架构
model_arch = 'unet';
enable_template_init = true;
template_sigma_sec = 0.045;

% enable_peak_guided_supervision = true 峰值引�? false关闭
% 过拟合降peak_guidance_alpha = 0.25 �?.20�?.10;
enable_peak_guided_supervision = true;
peak_guidance_alpha = 0.10;
peak_guidance_sigma_sec = 0.08;
% enable_dual_head_multitask = true;双头多任务，权重参数
% 过拟合降peak_loss_weight = 0.7; �?.4-0.5
enable_dual_head_multitask = false;

peak_loss_weight = 0.5;
% 时间注意力开�?
enable_temporal_attention = true;
% 自适应峰值检测开�?
enable_adaptive_peak_detection = true;
% 卡尔曼后处理开�?
enable_kalman_hr_smoothing = true;
% 高风险时序增强开关（false做基线对比）
enable_advanced_time_augmentation = true;


use_multi_file = true;
num_files_to_train = 3;
train_ratio = 0.70;
val_ratio = 0.15;
test_ratio = 0.15;


fprintf('=== ECG-supervised training mode ===\n');

%% Load paired radar + ECG files
x_cells = {};
y_cells = {};
ecg_cells = {};
pair_names = {};

if use_multi_file
    n_target_files = num_files_to_train;
else
    n_target_files = 1;
end

for file_idx = 1:n_target_files
    [radar_file, radar_path] = uigetfile({'*.bin'}, sprintf('Select radar file %d/%d', file_idx, n_target_files));
    if isequal(radar_file, 0)
        if isempty(x_cells)
            error('At least one paired radar+ECG file is required.');
        end
        break;
    end

    [ecg_file, ecg_path] = uigetfile({'*.mat;*.csv;*.txt', 'ECG files (*.mat,*.csv,*.txt)'}, ...
        sprintf('Select matching ECG file for %s', radar_file));
    if isequal(ecg_file, 0)
        error('ECG file selection cancelled for %s.', radar_file);
    end

    radar_path_full = fullfile(radar_path, radar_file);
    ecg_path_full = fullfile(ecg_path, ecg_file);

    x_ds = load_and_preprocess_radar(radar_path_full, fs_raw, downsample_factor, max_duration_minutes, ...
        heart_band_hz, enable_radar_despike, radar_despike_win_sec, radar_despike_k);
    [y_ref, ecg_meta, ecg_ds] = load_ecg_and_build_target_kd(ecg_path_full, fs_ds, numel(x_ds), heart_band_hz, valid_hr_bpm);

    % 自对�?- 使用鲁棒多段互相�?
    [x_ds, ecg_ds, y_ref, time_offset] = align_signals_by_xcorr(x_ds, ecg_ds, fs_ds, heart_band_hz, y_ref, ...
        align_num_segments, align_segment_sec, align_clip_k);
    
    L = min(numel(x_ds), numel(y_ref));
    x_ds = x_ds(1:L);
    y_ref = y_ref(1:L);
    ecg_ds = ecg_ds(1:L);
    if std(y_ref) < 0.05
        warning('ECG target is nearly flat after alignment. Check ECG file, peak detection, or alignment.');
    end

    x_cells{end+1,1} = x_ds(:); %#ok<SAGROW>
    y_cells{end+1,1} = y_ref(:); %#ok<SAGROW>
    ecg_cells{end+1,1} = ecg_ds(:); %#ok<SAGROW>
    pair_names{end+1,1} = sprintf('%s <-> %s', radar_file, ecg_file); %#ok<SAGROW>

    fprintf('Loaded pair %d: %s\n', file_idx, pair_names{end});
    fprintf('Duration: %.1f min, ECG peaks: %d, ECG fs: %.3f Hz\n', L/fs_ds/60, ecg_meta.n_peaks, ecg_meta.fs_ecg);
    fprintf('Time alignment offset: %.2f seconds\n', time_offset);
end

n_pairs = numel(x_cells);
if n_pairs == 0
    error('No valid paired data loaded.');
end

fprintf('Total loaded pairs: %d\n', n_pairs);

%% Split data before windowing (to avoid overlap leakage)
[x_tr_cells, y_tr_cells, ecg_tr_cells, x_va_cells, y_va_cells, ecg_va_cells, x_te_cells, y_te_cells, ecg_te_cells] = ...
    split_dataset_cells_kd(x_cells, y_cells, ecg_cells, [train_ratio, val_ratio, test_ratio]);

[Xtr, Ytr] = build_window_dataset(x_tr_cells, y_tr_cells, win_len, hop_len);
[Xva, Yva] = build_window_dataset(x_va_cells, y_va_cells, win_len, hop_len);
[Xte, Yte] = build_window_dataset(x_te_cells, y_te_cells, win_len, hop_len);

Xtr_ecg = build_window_dataset_single(ecg_tr_cells, win_len, hop_len);
Xva_ecg = build_window_dataset_single(ecg_va_cells, win_len, hop_len);
Xte_ecg = build_window_dataset_single(ecg_te_cells, win_len, hop_len);

fprintf('Windows before augmentation - train: %d, val: %d, test: %d\n', size(Xtr,2), size(Xva,2), size(Xte,2));
if size(Xtr, 2) < 50
    warning('Very few training windows (%d). Add more paired recordings.', size(Xtr, 2));
end

% Augment training set only (apply same transforms to radar/ECG/target)
[Xtr_aug, Xtr_ecg_aug, Ytr_aug] = augment_data_kd(Xtr, Xtr_ecg, Ytr, enable_advanced_time_augmentation);
fprintf('Training windows after augmentation: %d\n', size(Xtr_aug, 2));

if enable_peak_guided_supervision
    Ytr_aug = build_peak_guided_targets(Ytr_aug, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec, peak_guidance_alpha);
    Yva = build_peak_guided_targets(Yva, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec, peak_guidance_alpha);
    Yte = build_peak_guided_targets(Yte, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec, peak_guidance_alpha);
    fprintf('Peak-guided supervision enabled (alpha=%.2f, sigma=%.3fs).\n', peak_guidance_alpha, peak_guidance_sigma_sec);
end

Xtr4 = reshape(single(Xtr_aug), [win_len, 1, 1, size(Xtr_aug, 2)]);
Xva4 = reshape(single(Xva), [win_len, 1, 1, size(Xva, 2)]);
Xte4 = reshape(single(Xte), [win_len, 1, 1, size(Xte, 2)]);

Xtr_ecg4 = reshape(single(Xtr_ecg_aug), [win_len, 1, 1, size(Xtr_ecg_aug, 2)]);
Xva_ecg4 = reshape(single(Xva_ecg), [win_len, 1, 1, size(Xva_ecg, 2)]);
Xte_ecg4 = reshape(single(Xte_ecg), [win_len, 1, 1, size(Xte_ecg, 2)]);

if enable_dual_head_multitask
    Ytr_peak = build_peak_map_targets(Ytr_aug, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec);
    Yva_peak = build_peak_map_targets(Yva, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec);
    Yte_peak = build_peak_map_targets(Yte, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec);

    Ytr4 = cat(3, ...
        reshape(single(Ytr_aug), [win_len, 1, 1, size(Ytr_aug, 2)]), ...
        reshape(single(peak_loss_weight * Ytr_peak), [win_len, 1, 1, size(Ytr_peak, 2)]));
    Yva4 = cat(3, ...
        reshape(single(Yva), [win_len, 1, 1, size(Yva, 2)]), ...
        reshape(single(peak_loss_weight * Yva_peak), [win_len, 1, 1, size(Yva_peak, 2)]));
    Yte4 = cat(3, ...
        reshape(single(Yte), [win_len, 1, 1, size(Yte, 2)]), ...
        reshape(single(peak_loss_weight * Yte_peak), [win_len, 1, 1, size(Yte_peak, 2)]));

    fprintf('Dual-head multitask enabled (peak loss weight=%.2f).\n', peak_loss_weight);
else
    Ytr4 = reshape(single(Ytr_aug), [win_len, 1, 1, size(Ytr_aug, 2)]);
    Yva4 = reshape(single(Yva), [win_len, 1, 1, size(Yva, 2)]);
    Yte4 = reshape(single(Yte), [win_len, 1, 1, size(Yte, 2)]);
end

%% Train or load teacher + student with strict config check
model_config = struct();
model_config.win_len = win_len;
model_config.hop_len = hop_len;
model_config.fs_ds = fs_ds;
model_config.downsample_factor = downsample_factor;
model_config.heart_band_hz = heart_band_hz;
model_config.model_arch = model_arch;
model_config.enable_template_init = enable_template_init;
model_config.template_sigma_sec = template_sigma_sec;
model_config.enable_peak_guided_supervision = enable_peak_guided_supervision;
model_config.peak_guidance_alpha = peak_guidance_alpha;
model_config.peak_guidance_sigma_sec = peak_guidance_sigma_sec;
model_config.enable_dual_head_multitask = enable_dual_head_multitask;
model_config.peak_loss_weight = peak_loss_weight;
model_config.enable_temporal_attention = enable_temporal_attention;
model_config.enable_adaptive_peak_detection = enable_adaptive_peak_detection;
model_config.enable_kalman_hr_smoothing = enable_kalman_hr_smoothing;
model_config.enable_advanced_time_augmentation = enable_advanced_time_augmentation;
model_config.label_type = 'ecg_peak_train';
model_config.model_version = 11;
model_config.enable_kd = enable_kd;
model_config.kd_feature_weight = kd_feature_weight;
model_config.kd_response_weight = kd_response_weight;

if enable_kd && enable_dual_head_multitask
    warning('KD mode forces single-head output. Disabling dual-head multitask.');
    enable_dual_head_multitask = false;
end

teacher_config = model_config;
teacher_config.input_modality = 'ecg_teacher';
teacher_config.enable_kd = false;

% ---- Train or load teacher (ECG -> target) ----
need_retrain_teacher = true;
if exist(model_file_teacher, 'file') == 2
    S = load(model_file_teacher, 'teacher_net', 'teacher_config', 'teacher_train_info');
    if isfield(S, 'teacher_net') && isfield(S, 'teacher_config') && isequaln(S.teacher_config, teacher_config)
        teacher_net = S.teacher_net;
        teacher_train_info = S.teacher_train_info;
        need_retrain_teacher = false;
        fprintf('Loaded existing teacher model: %s\n', model_file_teacher);
    else
        fprintf('Existing teacher config mismatch. Re-training teacher.\n');
    end
end

if need_retrain_teacher
    [teacher_net, teacher_train_info] = train_heart_net(Xtr_ecg4, Ytr4, Xva_ecg4, Yva4, Xte_ecg4, Yte4, win_len, ...
        model_arch, enable_template_init, fs_ds, template_sigma_sec, ...
        enable_dual_head_multitask, enable_temporal_attention);
    save(model_file_teacher, 'teacher_net', 'teacher_config', 'teacher_train_info');
    fprintf('Trained and saved teacher model: %s\n', model_file_teacher);
end

% ---- Teacher quality gate (reason 3) ----
% Scheme A+B: trust test_corr if available; otherwise allow KD.
teacher_ok = true;
if exist('teacher_train_info', 'var')
    if isfield(teacher_train_info, 'test_corr') && isfinite(teacher_test_corr_min)
        if teacher_train_info.test_corr < teacher_test_corr_min
            teacher_ok = false;
            fprintf('Teacher test corr %.4f below threshold %.4f\n', teacher_train_info.test_corr, teacher_test_corr_min);
        end
    end
end

if ~teacher_ok
    warning('Teacher quality gate failed. Disabling KD for student training.');
    enable_kd = false;
    kd_feature_weight = 0.0;
    kd_response_weight = 0.0;
end

% Update student config after possible KD disable
model_config.enable_kd = enable_kd;
model_config.kd_feature_weight = kd_feature_weight;
model_config.kd_response_weight = kd_response_weight;

student_config = model_config;
student_config.input_modality = 'radar_student';

% ---- Train or load student (Radar -> target) with KD ----
need_retrain_student = true;
if exist(model_file_student, 'file') == 2
    S = load(model_file_student, 'student_net', 'student_config', 'student_train_info');
    if isfield(S, 'student_net') && isfield(S, 'student_config') && isequaln(S.student_config, student_config)
        student_net = S.student_net;
        student_train_info = S.student_train_info;
        need_retrain_student = false;
        fprintf('Loaded existing student model: %s\n', model_file_student);
    else
        fprintf('Existing student config mismatch. Re-training student.\n');
    end
end

if need_retrain_student
    if strcmpi(model_arch, 'unet')
        kd_feature_layer = kd_feature_layer_unet;
        kd_output_layer = 'conv_out';
    else
        kd_feature_layer = kd_feature_layer_resnet;
        kd_output_layer = 'conv_out2';
    end

    [student_net, student_train_info] = train_student_kd( ...
        Xtr4, Xtr_ecg4, Ytr4, Xva4, Xva_ecg4, Yva4, Xte4, Xte_ecg4, Yte4, ...
        teacher_net, model_arch, enable_template_init, fs_ds, template_sigma_sec, ...
        enable_temporal_attention, enable_kd, kd_feature_layer, kd_output_layer, ...
        kd_feature_weight, kd_response_weight, kd_use_global_pool);
    save(model_file_student, 'student_net', 'student_config', 'student_train_info', 'teacher_net', 'teacher_config');
    fprintf('Trained and saved student model: %s\n', model_file_student);
end

%% Inference on one hold-out sequence for visualization
if ~isempty(x_te_cells)
    x_eval = x_te_cells{1};
    y_eval = y_te_cells{1};
else
    x_eval = x_cells{1};
    y_eval = y_cells{1};
end
t_eval = (0:numel(x_eval)-1) / fs_ds;

[Xeval, idx_eval] = make_windows(x_eval, win_len, hop_len);
[Yeval, ~] = make_windows(y_eval, win_len, hop_len);
Xeval4 = reshape(single(Xeval), [win_len, 1, 1, size(Xeval, 2)]);
if enable_dual_head_multitask
    Yeval_peak = build_peak_map_targets(Yeval, fs_ds, valid_hr_bpm, peak_guidance_sigma_sec);
    Yeval4 = cat(3, ...
        reshape(single(Yeval), [win_len, 1, 1, size(Yeval, 2)]), ...
        reshape(single(peak_loss_weight * Yeval_peak), [win_len, 1, 1, size(Yeval_peak, 2)]));
else
    Yeval4 = reshape(single(Yeval), [win_len, 1, 1, size(Yeval, 2)]);
end

dlXeval = dlarray(single(Xeval4), 'SSCB');
Yhat4 = predict(student_net, dlXeval);
if isa(Yhat4, 'dlarray')
    Yhat4 = extractdata(Yhat4);
end
if enable_dual_head_multitask
    Yhat_wave4 = Yhat4(:, :, 1, :);
else
    Yhat_wave4 = Yhat4;
end

Yhat = squeeze(Yhat_wave4);
if isvector(Yhat)
    Yhat = reshape(Yhat, [win_len, 1]);
end

heart_dnn = overlap_add(Yhat, idx_eval, numel(x_eval));
% Robust normalization with clipping to suppress spikes
clip_val = 3;
heart_dnn = max(min(heart_dnn, clip_val), -clip_val);
heart_dnn = heart_dnn - median(heart_dnn);
heart_dnn = heart_dnn / (mad(heart_dnn, 1) + eps);

[b_bp, a_bp] = butter(3, heart_band_hz / (fs_ds / 2), 'bandpass');
heart_dnn = filtfilt(b_bp, a_bp, heart_dnn);
% Spike suppression (robust to isolated outliers)
m0 = median(heart_dnn);
s0 = mad(heart_dnn, 1) + eps;
thr = 6 * s0;
heart_dnn = min(max(heart_dnn, m0 - thr), m0 + thr);
heart_dnn = medfilt1(heart_dnn, 7);
heart_dnn = hampel(heart_dnn, round(0.3 * fs_ds), 3);
heart_dnn = heart_dnn / (std(heart_dnn) + eps);

metrics = evaluate_performance(heart_dnn, y_eval, Yeval4(:, :, 1, :), Yhat_wave4, fs_ds, heart_band_hz);

%% HR estimation for predicted and ECG-reference signals
[hr_pred_raw, t_hr_pred, pks_pred, locs_pred] = estimate_hr_from_signal(heart_dnn, fs_ds, valid_hr_bpm, enable_adaptive_peak_detection);
[hr_ref, t_hr_ref, ~, ~] = estimate_hr_from_signal(y_eval, fs_ds, valid_hr_bpm, true);

if enable_kalman_hr_smoothing
    hr_pred = kalman_smooth_hr(hr_pred_raw, t_hr_pred);
else
    hr_pred = hr_pred_raw;
end

if ~isempty(hr_pred)
    fprintf('\n=== DNN Heart-Rate (hold-out) ===\n');
    fprintf('Mean HR: %.1f BPM, Std: %.1f BPM, Median: %.1f BPM\n', mean(hr_pred), std(hr_pred), median(hr_pred));
    fprintf('Range: %.1f - %.1f BPM, Beats: %d\n', min(hr_pred), max(hr_pred), numel(hr_pred));
else
    fprintf('\nNo valid HR detected from DNN signal on hold-out sequence.\n');
end

if ~isempty(hr_ref)
    fprintf('ECG-ref Mean HR: %.1f BPM, Std: %.1f BPM\n', mean(hr_ref), std(hr_ref));
end

if ~isempty(hr_pred) && ~isempty(hr_ref)
    [hr_mae_bpm, hr_rmse_bpm] = compare_hr_series(t_hr_pred, hr_pred, t_hr_ref, hr_ref);
    fprintf('HR MAE vs ECG-ref: %.2f BPM, RMSE: %.2f BPM\n', hr_mae_bpm, hr_rmse_bpm);
end

fprintf('\n=== Signal-Level Metrics (hold-out) ===\n');
fprintf('MAE:  %.4f\n', metrics.mae);
fprintf('RMSE: %.4f\n', metrics.rmse);
fprintf('SNR:  %.2f dB\n', metrics.snr);
fprintf('Correlation: %.4f\n', metrics.corr);
fprintf('Spectral RMSE: %.4f\n', metrics.spectral_rmse);
fprintf('HR-band Energy RelErr: %.4f\n', metrics.band_energy_relerr);

%% Visualization
figure('Position', [80, 60, 1600, 1200]);

subplot(6,1,1);
plot(t_eval/60, x_eval, 'k-');
title('Input Radar Signal'); xlabel('Time (min)'); ylabel('Amplitude'); grid on;

subplot(6,1,2);
plot(t_eval/60, y_eval, 'b-');
title('ECG-derived Target'); xlabel('Time (min)'); ylabel('Amplitude'); grid on;

subplot(6,1,3);
plot(t_eval/60, heart_dnn, 'g-'); hold on;
if ~isempty(locs_pred)
    plot(t_eval(locs_pred)/60, pks_pred, 'ro', 'MarkerSize', 4);
end
title('Network Output + Peaks'); xlabel('Time (min)'); ylabel('Amplitude'); grid on;

subplot(6,1,4);
if ~isempty(hr_pred)
    plot(t_hr_pred/60, hr_pred_raw, 'b-o', 'MarkerSize', 3); hold on;
    if enable_kalman_hr_smoothing && numel(hr_pred_raw) == numel(hr_pred)
        plot(t_hr_pred/60, hr_pred, 'r-', 'LineWidth', 1.8);
        yline(mean(hr_pred), 'k--', sprintf('Mean %.1f BPM', mean(hr_pred)));
        legend('Raw HR', 'Kalman HR', 'Mean', 'Location', 'best');
    else
        hr_pred_s = movmean(hr_pred, min(7, numel(hr_pred)));
        plot(t_hr_pred/60, hr_pred_s, 'r-', 'LineWidth', 1.8);
        yline(mean(hr_pred), 'k--', sprintf('Mean %.1f BPM', mean(hr_pred)));
        legend('Raw HR', 'Smoothed HR', 'Mean', 'Location', 'best');
    end
    title('Instantaneous HR (Predicted)'); xlabel('Time (min)'); ylabel('HR (BPM)'); grid on;
    ylim([valid_hr_bpm(1)-5, valid_hr_bpm(2)+5]);
else
    text(0.5, 0.5, 'No valid predicted HR points', 'HorizontalAlignment', 'center');
    axis off;
end

subplot(6,1,5);
if ~exist('student_train_info', 'var')
    student_train_info = struct();
end
train_info = student_train_info;
if isfield(train_info, 'train_loss') && ~isempty(train_info.train_loss)
    yyaxis left;
    plot(train_info.train_loss, 'b-', 'LineWidth', 1.3);
    ylabel('Training Loss'); xlabel('Iteration');
    yyaxis right;
    plot(train_info.val_loss, 'r-', 'LineWidth', 1.3);
    ylabel('Validation Loss');
    title('Training History'); grid on; legend('Train', 'Validation');
else
    text(0.5, 0.5, 'Training history unavailable (loaded old model).', 'HorizontalAlignment', 'center');
    axis off;
end

subplot(6,1,6);
N = numel(heart_dnn);
F = (0:floor(N/2)) * fs_ds / N;
P = abs(fft(heart_dnn));
P = P(1:floor(N/2)+1).^2;
sel = F >= 0.4 & F <= 3.2;
plot(F(sel)*60, P(sel), 'm-', 'LineWidth', 1.2);
title('Spectrum'); xlabel('Frequency (BPM)'); ylabel('Power'); grid on; xlim([24, 192]);

%% Local functions
function x_ds = load_and_preprocess_radar(file_path, fs_raw, downsample_factor, max_duration_minutes, ...
    heart_band_hz, enable_radar_despike, radar_despike_win_sec, radar_despike_k)
    max_samples = max_duration_minutes * 60 * fs_raw;
    [i_raw, q_raw, ~] = read_iq_bin(file_path, max_samples);
    if isempty(i_raw) || isempty(q_raw)
        error('Empty or unreadable radar file: %s', file_path);
    end

    iq = complex(i_raw, q_raw);
    mag = abs(iq);
    mag = fillmissing(mag, 'linear');
    mag = mag - mean(mag);

    [b_lp, a_lp] = butter(4, 28 / (fs_raw / 2), 'low');
    mag_lp = filtfilt(b_lp, a_lp, mag);
    mag_ds = downsample(mag_lp, downsample_factor);

    ph = unwrap(angle(iq));
    ph = detrend(ph);
    ph = ph - mean(ph);
    ph = fillmissing(ph, 'linear');
    ph_ds = downsample(ph, downsample_factor);

    if downsample_factor > 0
        fs_ds_local = fs_raw / downsample_factor;
    else
        fs_ds_local = fs_raw;
    end

    [b_bp, a_bp] = butter(2, heart_band_hz / (fs_ds_local / 2), 'bandpass');
    mag_ds = fillmissing(mag_ds, 'linear');
    ph_ds = fillmissing(ph_ds, 'linear');
    mag_bp = filtfilt(b_bp, a_bp, mag_ds);
    ph_bp = filtfilt(b_bp, a_bp, ph_ds);

    if std(ph_bp) > std(mag_bp)
        x_ds = ph_bp;
    else
        x_ds = mag_bp;
    end

    if nargin >= 6 && enable_radar_despike
        x_ds = suppress_outliers_moving(x_ds, fs_ds_local, radar_despike_win_sec, radar_despike_k);
    end
    x_ds = (x_ds - mean(x_ds)) / (std(x_ds) + eps);
end

function [i_raw, q_raw, dbg] = read_iq_bin(file_path, max_samples)
    formats = {'float32', 'int16', 'uint16'};
    best = [];
    best_dbg = struct('format', '', 'scale', 1, 'max_abs', NaN);
    for k = 1:numel(formats)
        fmt = formats{k};
        [i_try, q_try] = read_interleaved_iq(file_path, fmt, max_samples);
        if isempty(i_try) || isempty(q_try)
            continue;
        end
        max_abs = max(abs([i_try(:); q_try(:)]));
        if ~isfinite(max_abs) || max_abs == 0
            continue;
        end
        score = max_abs;
        if max_abs > 1e6
            score = score * 1e6;
        end
        if isempty(best) || score < best.score
            best.score = score;
            best.i = i_try;
            best.q = q_try;
            best_dbg = struct('format', fmt, 'scale', 1, 'max_abs', max_abs);
        end
    end

    if isempty(best)
        i_raw = [];
        q_raw = [];
        dbg = best_dbg;
        return;
    end

    i_raw = double(best.i(:).');
    q_raw = double(best.q(:).');

    max_abs = max(abs([i_raw, q_raw]));
    if isfinite(max_abs) && max_abs > 0
        i_raw = i_raw / max_abs;
        q_raw = q_raw / max_abs;
        best_dbg.scale = max_abs;
        best_dbg.max_abs = max_abs;
    end

    i_raw(~isfinite(i_raw)) = NaN;
    q_raw(~isfinite(q_raw)) = NaN;
    i_raw = fillmissing(i_raw, 'linear');
    q_raw = fillmissing(q_raw, 'linear');
    i_raw(~isfinite(i_raw)) = 0;
    q_raw(~isfinite(q_raw)) = 0;
    dbg = best_dbg;
end

function [i_raw, q_raw] = read_interleaved_iq(file_path, fmt, max_samples)
    i_raw = [];
    q_raw = [];
    fileID = fopen(file_path, 'r');
    if fileID < 0
        return;
    end
    if isfinite(max_samples) && max_samples > 0
        count = 2 * max_samples;
        data = fread(fileID, count, ['*' fmt]);
    else
        data = fread(fileID, inf, ['*' fmt]);
    end
    fclose(fileID);
    if isempty(data)
        return;
    end
    if mod(numel(data), 2) ~= 0
        data = data(1:end-1);
    end
    i_raw = data(1:2:end);
    q_raw = data(2:2:end);
end

function x = suppress_outliers_moving(x, fs, win_sec, k)
    x = x(:);
    if isempty(x) || ~isfinite(fs) || fs <= 0
        return;
    end
    if nargin < 3 || isempty(win_sec)
        win_sec = 0.6;
    end
    if nargin < 4 || isempty(k)
        k = 6;
    end
    win = max(5, round(win_sec * fs));
    if numel(x) < win || win < 3
        return;
    end
    med = movmedian(x, win, 'omitnan');
    madv = movmad(x, win, 1, 'omitnan');
    thresh = k * madv + 1e-6;
    idx = abs(x - med) > thresh;
    x(idx) = med(idx);
end

function [y_ref, meta, ecg_ds] = load_ecg_and_build_target_kd(ecg_path, fs_ds, target_len, heart_band_hz, valid_hr_bpm)
    [ecg, fs_ecg] = load_ecg_signal(ecg_path);
    if isempty(ecg)
        error('No ECG samples found in %s', ecg_path);
    end
    if ~isfinite(fs_ecg) || fs_ecg <= 0
        fprintf('Warning: Could not automatically infer ECG sampling rate from %s\n', ecg_path);
        fprintf('ECG file has %d samples\n', numel(ecg));
        fs_ecg = input('Enter ECG sampling rate (Hz): ');
        if isempty(fs_ecg) || fs_ecg <= 0
            error('Invalid ECG sampling rate.');
        end
    else
        fprintf('Inferred ECG sampling rate: %.3f Hz\n', fs_ecg);
    end

    ecg = ecg(:);
    ecg = fillmissing(ecg, 'linear');
    ecg = ecg - mean(ecg);

    % Bandpass ECG for teacher input
    if fs_ecg > 55
        [b_ecg, a_ecg] = butter(3, [5, 20] / (fs_ecg / 2), 'bandpass');
        ecg_bp = filtfilt(b_ecg, a_ecg, ecg);
    else
        ecg_bp = ecg;
    end

    [~, locs] = pan_tompkins_process(ecg_bp, fs_ecg, valid_hr_bpm);
    rr_cv = inf;
    if numel(locs) >= 5
        rr = diff(locs) / fs_ecg;
        rr_cv = std(rr) / (mean(rr) + eps);
    end
    if numel(locs) < 3 || rr_cv < 0.02
        if rr_cv < 0.02
            fprintf('Pan-Tompkins peaks too regular (RR CV=%.3f). Falling back.\n', rr_cv);
        end
        env = movmean(ecg_bp.^2, max(1, round(0.08 * fs_ecg)));
        min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs_ecg));
        prom = max(0.2 * std(env), 1.0 * mad(env, 1));
        [~, locs] = findpeaks(env, 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', prom);
        if numel(locs) < 5
            [~, locs] = findpeaks(abs(ecg_bp), 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', prom * 0.5);
        end
    end
    if numel(locs) < 3
        warning('Insufficient ECG peaks in %s. Falling back to ECG band-limited waveform.', ecg_path);
        t_ecg = (0:numel(ecg_bp)-1) / fs_ecg;
        t_ds = (0:target_len-1) / fs_ds;
        y_interp = interp1(t_ecg, ecg_bp, t_ds, 'linear', 'extrap');
        y_ref = y_interp(:);
        [b_bp, a_bp] = butter(2, heart_band_hz / (fs_ds / 2), 'bandpass');
        y_ref = filtfilt(b_bp, a_bp, y_ref);
        y_ref = y_ref / (std(y_ref) + eps);

        ecg_ds = y_ref;
        meta = struct('n_peaks', 0, 'fs_ecg', fs_ecg);
        return;
    end

    peak_times = (locs - 1) / fs_ecg;
    peak_ds = unique(max(1, min(target_len, round(peak_times * fs_ds) + 1)));

    pulse = zeros(target_len, 1);
    pulse(peak_ds) = 1;

    sigma = max(1, round(0.04 * fs_ds));
    k = (-round(0.20 * fs_ds):round(0.20 * fs_ds))';
    g = exp(-0.5 * (k / sigma).^2);
    g = g / (sum(g) + eps);
    y_ref = conv(pulse, g, 'same');

    [b_bp, a_bp] = butter(2, heart_band_hz / (fs_ds / 2), 'bandpass');
    y_ref = filtfilt(b_bp, a_bp, y_ref);
    y_ref = y_ref - mean(y_ref);
    y_ref = y_ref / (std(y_ref) + eps);

    % Build teacher ECG input at fs_ds
    t_ecg = (0:numel(ecg_bp)-1) / fs_ecg;
    t_ds = (0:target_len-1) / fs_ds;
    ecg_ds = interp1(t_ecg, ecg_bp, t_ds, 'linear', 'extrap');
    [b_bp2, a_bp2] = butter(2, heart_band_hz / (fs_ds / 2), 'bandpass');
    ecg_ds = filtfilt(b_bp2, a_bp2, ecg_ds);
    ecg_ds = ecg_ds - mean(ecg_ds);
    ecg_ds = ecg_ds / (std(ecg_ds) + eps);

    meta = struct('n_peaks', numel(peak_ds), 'fs_ecg', fs_ecg);
end

function [y_ref, meta] = load_ecg_and_build_target(ecg_path, fs_ds, target_len, heart_band_hz, valid_hr_bpm)
    [ecg, fs_ecg] = load_ecg_signal(ecg_path);
    if isempty(ecg)
        error('No ECG samples found in %s', ecg_path);
    end
    if ~isfinite(fs_ecg) || fs_ecg <= 0
        fprintf('Warning: Could not automatically infer ECG sampling rate from %s\n', ecg_path);
        fprintf('ECG file has %d samples\n', numel(ecg));
        fs_ecg = input('Enter ECG sampling rate (Hz): ');
        if isempty(fs_ecg) || fs_ecg <= 0
            error('Invalid ECG sampling rate.');
        end
    else
        fprintf('Inferred ECG sampling rate: %.3f Hz\n', fs_ecg);
    end

    ecg = ecg(:);
    ecg = fillmissing(ecg, 'linear');
    ecg = ecg - mean(ecg);

    if fs_ecg > 55
        [b_ecg, a_ecg] = butter(3, [5, 20] / (fs_ecg / 2), 'bandpass');
        ecg_bp = filtfilt(b_ecg, a_ecg, ecg);
    else
        ecg_bp = ecg;
    end

    [~, locs] = pan_tompkins_process(ecg_bp, fs_ecg, valid_hr_bpm);
    rr_cv = inf;
    if numel(locs) >= 5
        rr = diff(locs) / fs_ecg;
        rr_cv = std(rr) / (mean(rr) + eps);
    end
    if numel(locs) < 3 || rr_cv < 0.02
        if rr_cv < 0.02
            fprintf('Pan-Tompkins peaks too regular (RR CV=%.3f). Falling back.\n', rr_cv);
        end
        env = movmean(ecg_bp.^2, max(1, round(0.08 * fs_ecg)));
        min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs_ecg));
        prom = max(0.2 * std(env), 1.0 * mad(env, 1));
        [~, locs] = findpeaks(env, 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', prom);
        if numel(locs) < 5
            [~, locs] = findpeaks(abs(ecg_bp), 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', prom * 0.5);
        end
    end
    if numel(locs) < 3
        warning('Insufficient ECG peaks in %s. Falling back to ECG band-limited waveform.', ecg_path);
        t_ecg = (0:numel(ecg_bp)-1) / fs_ecg;
        t_ds = (0:target_len-1) / fs_ds;
        y_interp = interp1(t_ecg, ecg_bp, t_ds, 'linear', 'extrap');
        y_ref = y_interp(:);
        [b_bp, a_bp] = butter(2, heart_band_hz / (fs_ds / 2), 'bandpass');
        y_ref = filtfilt(b_bp, a_bp, y_ref);
        y_ref = y_ref / (std(y_ref) + eps);
        meta = struct('n_peaks', 0, 'fs_ecg', fs_ecg);
        return;
    end

    peak_times = (locs - 1) / fs_ecg;
    peak_ds = unique(max(1, min(target_len, round(peak_times * fs_ds) + 1)));

    pulse = zeros(target_len, 1);
    pulse(peak_ds) = 1;

    sigma = max(1, round(0.04 * fs_ds));
    k = (-round(0.20 * fs_ds):round(0.20 * fs_ds))';
    g = exp(-0.5 * (k / sigma).^2);
    g = g / (sum(g) + eps);
    y_ref = conv(pulse, g, 'same');

    [b_bp, a_bp] = butter(2, heart_band_hz / (fs_ds / 2), 'bandpass');
    y_ref = filtfilt(b_bp, a_bp, y_ref);
    y_ref = y_ref - mean(y_ref);
    y_ref = y_ref / (std(y_ref) + eps);

    meta = struct('n_peaks', numel(peak_ds), 'fs_ecg', fs_ecg);
end

function [mwi, locs] = pan_tompkins_process(ecg, fs_ecg, valid_hr_bpm)
    ecg = double(ecg(:));
    ecg = ecg - mean(ecg);
    ecg = fillmissing(ecg, 'linear');

    % Bandpass (QRS band) with safe bounds
    if fs_ecg > 30
        lowcut = min(5, 0.1 * fs_ecg);
        highcut = min(20, 0.45 * fs_ecg);
        if highcut > lowcut && highcut > 0
            wn = [lowcut, highcut] / (fs_ecg / 2);
            wn = max(min(wn, 0.99), 0.01);
            if wn(2) > wn(1)
                [b, a] = butter(3, wn, 'bandpass');
                ecg_f = filtfilt(b, a, ecg);
            else
                ecg_f = ecg;
            end
        else
            ecg_f = ecg;
        end
    else
        ecg_f = ecg;
    end

    % Derivative + squaring
    d = [0; diff(ecg_f)];
    d2 = d .^ 2;
    
    % Moving window integration (~150 ms)
    win = max(1, round(0.15 * fs_ecg));
    mwi = movmean(d2, win);

    % Peak detection
    min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs_ecg));
    thr = max(mean(mwi) + 0.5 * std(mwi), 0.3 * max(mwi));
    [~, locs] = findpeaks(mwi, 'MinPeakDistance', min_peak_dist, 'MinPeakHeight', thr);
    if numel(locs) < 3
        thr2 = max(mean(mwi) + 0.25 * std(mwi), 0.2 * max(mwi));
        [~, locs] = findpeaks(mwi, 'MinPeakDistance', min_peak_dist, 'MinPeakHeight', thr2);
    end
end

function [ecg, fs_ecg] = load_ecg_signal(ecg_path)
    ecg = [];
    fs_ecg = NaN;
    [~, ~, ext] = fileparts(ecg_path);
    ext = lower(ext);

    switch ext
        case '.mat'
            S = load(ecg_path);
            fns = fieldnames(S);

            fs_candidates = {'fs', 'Fs', 'FS', 'ecg_fs', 'sampling_rate', 'sample_rate'};
            for i = 1:numel(fs_candidates)
                if isfield(S, fs_candidates{i})
                    fs_tmp = S.(fs_candidates{i});
                    if isnumeric(fs_tmp) && isscalar(fs_tmp)
                        fs_ecg = double(fs_tmp);
                        break;
                    end
                end
            end

            for i = 1:numel(fns)
                v = S.(fns{i});
                if isnumeric(v) && isvector(v) && numel(v) > 100
                    ecg = double(v(:));
                    break;
                end
            end

        case {'.csv', '.txt'}
            % Handle "timestamp: ADC HeartRate4sAverage HeartRate30sAverage"
            txt = readlines(ecg_path);
            % Convert string array to cell array for compatibility
            if isstring(txt)
                txt = cellstr(txt);
            end
            txt = txt(~cellfun(@isempty, txt));
            if isempty(txt)
                error('Unsupported ECG text format: %s', ecg_path);
            end
            nums = cellfun(@(s) regexp(s, '[-+]?\d*\.?\d+', 'match'), txt, 'UniformOutput', false);
            nums = nums(~cellfun(@isempty, nums));
            if isempty(nums)
                error('Unsupported ECG text format: %s', ecg_path);
            end
            rows = cellfun(@(c) str2double(c(:)'), nums, 'UniformOutput', false);
            maxn = max(cellfun(@numel, rows));
            vals = NaN(numel(rows), maxn);
            for ii = 1:numel(rows)
                vals(ii, 1:numel(rows{ii})) = rows{ii};
            end
            if size(vals,2) >= 2
                t = vals(:,1);
                ecg = vals(:,2); % ADC column
                fs_inf = infer_fs_from_timestamp_series(t);
                if isfinite(fs_inf)
                    fs_ecg = fs_inf;
                end
            else
                M = readmatrix(ecg_path);
                if isempty(M) || ~isnumeric(M)
                    error('Unsupported ECG text format: %s', ecg_path);
                end
                ecg = double(M(:,1));
            end

        otherwise
            error('Unsupported ECG file extension: %s', ext);
    end

    ecg = ecg(:);
    ecg = ecg(isfinite(ecg));
end

function fs = infer_fs_from_time(t)
    fs = NaN;
    if isempty(t) || numel(t) < 10
        return;
    end
    dt = diff(t(:));
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt)
        return;
    end
    med_dt = median(dt);
    if med_dt <= 0
        return;
    end
    if med_dt > 1
        med_dt = med_dt / 1000;
    end
    fs = 1 / med_dt;
end

function fs = infer_fs_from_timestamp_series(t)
    fs = NaN;
    t = t(:);
    if numel(t) < 10
        return;
    end
    dt = diff(t);
    if all(dt == 0)
        return;
    end
    change_idx = [1; find(dt ~= 0) + 1];
    if numel(change_idx) >= 2
        tchg = t(change_idx);
        dtchg = diff(tchg);
        ns = diff([change_idx; numel(t)+1]);
        ok = dtchg > 0 & isfinite(dtchg);
        if any(ok)
            fs = median(ns(ok) ./ dtchg(ok));
            return;
        end
    end
    tu = unique(t);
    fs = infer_fs_from_time(tu);
end

function [x_aligned, y_aligned, y2_aligned, time_offset_sec] = align_signals_by_xcorr(x, y, fs, heart_band_hz, y2, ...
    num_segments, segment_sec, clip_k)
    x = x(:);
    y = y(:);
    if nargin < 5
        y2 = [];
    else
        y2 = y2(:);
    end
    if nargin < 6 || isempty(num_segments)
        num_segments = 3;
    end
    if nargin < 7 || isempty(segment_sec)
        segment_sec = 20;
    end
    if nargin < 8 || isempty(clip_k)
        clip_k = 3.0;
    end
    
    [b_bp, a_bp] = butter(3, heart_band_hz / (fs / 2), 'bandpass');
    x_bp = filtfilt(b_bp, a_bp, x - mean(x));
    y_bp = filtfilt(b_bp, a_bp, y - mean(y));

    x_bp = robust_clip(x_bp, clip_k);
    y_bp = robust_clip(y_bp, clip_k);
    
    L = min(numel(x_bp), numel(y_bp));
    seg_len = max(8, round(segment_sec * fs));
    seg_len = min(seg_len, L);
    if seg_len < 8
        seg_len = L;
        num_segments = 1;
    end
    
    lags_all = [];
    max_lag = round(10 * fs);
    for s = 1:num_segments
        start_idx = 1 + round((s-1) * (L - seg_len) / max(1, num_segments-1));
        stop_idx = min(L, start_idx + seg_len - 1);
        x_seg = x_bp(start_idx:stop_idx);
        y_seg = y_bp(start_idx:stop_idx);
        [xcorr_vals, lags] = xcorr(x_seg, y_seg, max_lag, 'coeff');
        [~, max_idx] = max(xcorr_vals);
        lags_all(end+1,1) = lags(max_idx);
    end
    
    best_lag = round(median(lags_all));
    time_offset_sec = best_lag / fs;
    
    if best_lag > 0
        y_aligned = y(best_lag+1:end);
        x_aligned = x(1:numel(y_aligned));
        if ~isempty(y2)
            y2_aligned = y2(best_lag+1:end);
            y2_aligned = y2_aligned(1:numel(y_aligned));
        else
            y2_aligned = [];
        end
    elseif best_lag < 0
        x_aligned = x(-best_lag+1:end);
        y_aligned = y(1:numel(x_aligned));
        if ~isempty(y2)
            y2_aligned = y2(1:numel(x_aligned));
        else
            y2_aligned = [];
        end
    else
        x_aligned = x;
        y_aligned = y;
        y2_aligned = y2;
    end
    
    fprintf('  Cross-correlation peak at lag=%d samples (%.2f sec)\n', best_lag, time_offset_sec);
end























































function [x_tr_cells, y_tr_cells, ecg_tr_cells, x_va_cells, y_va_cells, ecg_va_cells, x_te_cells, y_te_cells, ecg_te_cells] = split_dataset_cells_kd(x_cells, y_cells, ecg_cells, ratios)
    n = numel(x_cells);
    if n ~= numel(y_cells) || n ~= numel(ecg_cells)
        error('Input cell size mismatch.');
    end

    if n >= 3
        idx = randperm(n);
        n_tr = max(1, floor(ratios(1) * n));
        n_va = max(1, floor(ratios(2) * n));
        n_te = n - n_tr - n_va;
        if n_te < 1
            n_te = 1;
            n_tr = max(1, n_tr - 1);
        end

        tr_idx = idx(1:n_tr);
        va_idx = idx(n_tr+1:n_tr+n_va);
        te_idx = idx(n_tr+n_va+1:end);

        x_tr_cells = x_cells(tr_idx); y_tr_cells = y_cells(tr_idx); ecg_tr_cells = ecg_cells(tr_idx);
        x_va_cells = x_cells(va_idx); y_va_cells = y_cells(va_idx); ecg_va_cells = ecg_cells(va_idx);
        x_te_cells = x_cells(te_idx); y_te_cells = y_cells(te_idx); ecg_te_cells = ecg_cells(te_idx);
        return;
    end

    if n == 2
        idx = randperm(2);
        tr_idx = idx(1);
        hold_idx = idx(2);

        x_tr_cells = {x_cells{tr_idx}(:)};
        y_tr_cells = {y_cells{tr_idx}(:)};
        ecg_tr_cells = {ecg_cells{tr_idx}(:)};

        x_va_cells = {x_cells{hold_idx}(:)};
        y_va_cells = {y_cells{hold_idx}(:)};
        ecg_va_cells = {ecg_cells{hold_idx}(:)};

        % Avoid val/test overlap when only two files are available
        x_te_cells = {};
        y_te_cells = {};
        ecg_te_cells = {};
        return;
    end

    x_tr_cells = {x_cells{1}(:)}; y_tr_cells = {y_cells{1}(:)}; ecg_tr_cells = {ecg_cells{1}(:)};
    x_va_cells = x_tr_cells; y_va_cells = y_tr_cells; ecg_va_cells = ecg_tr_cells;
    x_te_cells = x_tr_cells; y_te_cells = y_tr_cells; ecg_te_cells = ecg_tr_cells;
end


function [x_tr_cells, y_tr_cells, x_va_cells, y_va_cells, x_te_cells, y_te_cells] = split_dataset_cells(x_cells, y_cells, ratios)
    n = numel(x_cells);
    if n ~= numel(y_cells)
        error('Input cell size mismatch.');
    end

    if n >= 3
        idx = randperm(n);
        n_tr = max(1, floor(ratios(1) * n));
        n_va = max(1, floor(ratios(2) * n));
        n_te = n - n_tr - n_va;
        if n_te < 1
            n_te = 1;
            n_tr = max(1, n_tr - 1);
        end

        tr_idx = idx(1:n_tr);
        va_idx = idx(n_tr+1:n_tr+n_va);
        te_idx = idx(n_tr+n_va+1:end);

        x_tr_cells = x_cells(tr_idx); y_tr_cells = y_cells(tr_idx);
        x_va_cells = x_cells(va_idx); y_va_cells = y_cells(va_idx);
        x_te_cells = x_cells(te_idx); y_te_cells = y_cells(te_idx);
        return;
    end

    if n == 2
        idx = randperm(2);
        tr_idx = idx(1);
        hold_idx = idx(2);

        x_tr_cells = {x_cells{tr_idx}(:)};
        y_tr_cells = {y_cells{tr_idx}(:)};

        x_hold = x_cells{hold_idx}(:);
        y_hold = y_cells{hold_idx}(:);
        L = min(numel(x_hold), numel(y_hold));
        x_hold = x_hold(1:L);
        y_hold = y_hold(1:L);

        val_share = ratios(2) / (ratios(2) + ratios(3) + eps);
        i_cut = max(2, min(L - 2, floor(val_share * L)));

        x_va_cells = {x_hold(1:i_cut)};
        y_va_cells = {y_hold(1:i_cut)};
        x_te_cells = {x_hold(i_cut+1:end)};
        y_te_cells = {y_hold(i_cut+1:end)};
        return;
    end

    x = x_cells{1}(:);
    y = y_cells{1}(:);
    L = min(numel(x), numel(y));
    x = x(1:L);
    y = y(1:L);

    i1 = max(2, floor(ratios(1) * L));
    i2 = max(i1 + 2, floor((ratios(1) + ratios(2)) * L));
    i2 = min(i2, L - 1);

    x_tr_cells = {x(1:i1)};
    y_tr_cells = {y(1:i1)};
    x_va_cells = {x(i1+1:i2)};
    y_va_cells = {y(i1+1:i2)};
    x_te_cells = {x(i2+1:end)};
    y_te_cells = {y(i2+1:end)};
end


function X = build_window_dataset_single(x_cells, win_len, hop_len)
    X = [];
    for i = 1:numel(x_cells)
        x = x_cells{i};
        [Wx, ~] = make_windows(x, win_len, hop_len);
        X = [X, Wx]; %#ok<AGROW>
    end
end

function [X, Y] = build_window_dataset(x_cells, y_cells, win_len, hop_len)
    X = [];
    Y = [];
    for i = 1:numel(x_cells)
        x = x_cells{i}(:);
        y = y_cells{i}(:);
        L = min(numel(x), numel(y));
        if L < win_len
            continue;
        end
        [Xi, ~] = make_windows(x(1:L), win_len, hop_len);
        [Yi, ~] = make_windows(y(1:L), win_len, hop_len);
        X = [X, Xi]; %#ok<AGROW>
        Y = [Y, Yi]; %#ok<AGROW>
    end
end

function [student_net, train_info] = train_student_kd(Xtr4, Xtr_ecg4, Ytr4, Xva4, Xva_ecg4, Yva4, Xte4, Xte_ecg4, Yte4, ...
    teacher_net, model_arch, enable_template_init, fs_ds, template_sigma_sec, enable_temporal_attention, ...
    enable_kd, kd_feature_layer, kd_output_layer, kd_feature_weight, kd_response_weight, kd_use_global_pool)
    if strcmpi(model_arch, 'unet')
        lgraph = create_unet_lgraph(size(Xtr4, 1), fs_ds, enable_template_init, template_sigma_sec, 1, enable_temporal_attention);
    else
        lgraph = create_resnet_lgraph(size(Xtr4, 1), fs_ds, enable_template_init, template_sigma_sec, 1, enable_temporal_attention);
    end

    student_net = prepare_dlnet_from_lgraph(lgraph);
    teacher_dlnet = make_dlnet_from_trained(teacher_net);

    max_epochs = 30;
    mini_batch = 64;
    learn_rate = 5e-4;

    warmup_epochs = 15;
    kd_feat_max = kd_feature_weight;
    kd_resp_max = kd_response_weight;
    lr_base = learn_rate;

    num_obs = size(Xtr4, 4);
    num_iter = max(1, ceil(num_obs / mini_batch));

    trailingAvg = [];
    trailingAvgSq = [];
    iteration = 0;

    train_loss = zeros(max_epochs, 1);
    val_loss = zeros(max_epochs, 1);
    best_val = inf;
    best_epoch = 0;
    best_net = student_net;
    patience = 5;
    patience_counter = 0;

    for epoch = 1:max_epochs
        if epoch > 25
            lr = lr_base * 0.1;
        elseif epoch > 15
            lr = lr_base * 0.3;
        else
            % Keep LR gentler during KD warmup
            ramp_lr = min(1.0, epoch / max(1, warmup_epochs));
            lr = lr_base * (0.5 + 0.5 * ramp_lr);
        end

        if enable_kd
            ramp = min(1.0, epoch / max(1, warmup_epochs));
            kd_feat_w = kd_feat_max * ramp;
            kd_resp_w = kd_resp_max * ramp;
        else
            kd_feat_w = 0;
            kd_resp_w = 0;
        end

        idx = randperm(num_obs);
        loss_epoch = 0;
        for i = 1:mini_batch:num_obs
            iteration = iteration + 1;
            batch = idx(i:min(i+mini_batch-1, num_obs));

            Xb = Xtr4(:, :, :, batch);
            Xb_ecg = Xtr_ecg4(:, :, :, batch);
            Yb = Ytr4(:, :, :, batch);

            dlX = dlarray(single(Xb), 'SSCB');
            dlXecg = dlarray(single(Xb_ecg), 'SSCB');
            dlY = dlarray(single(Yb), 'SSCB');

            [gradients, loss] = dlfeval(@modelGradientsKD, student_net, teacher_dlnet, dlX, dlXecg, dlY, ...
                enable_kd, kd_feature_layer, kd_output_layer, kd_feat_w, kd_resp_w, kd_use_global_pool);

            [student_net, trailingAvg, trailingAvgSq] = adamupdate(student_net, gradients, trailingAvg, trailingAvgSq, iteration, lr);
            loss_epoch = loss_epoch + double(gather(extractdata(loss)));
        end
        train_loss(epoch) = loss_epoch / num_iter;

        val_loss(epoch) = evaluate_kd_loss(student_net, teacher_dlnet, Xva4, Xva_ecg4, Yva4, mini_batch, ...
            enable_kd, kd_feature_layer, kd_output_layer, kd_feat_w, kd_resp_w, kd_use_global_pool);

        fprintf('Epoch %d/%d - train loss %.4f - val loss %.4f\n', epoch, max_epochs, train_loss(epoch), val_loss(epoch));

        if val_loss(epoch) < best_val - 1e-4
            best_val = val_loss(epoch);
            best_epoch = epoch;
            best_net = student_net;
            patience_counter = 0;
        else
            patience_counter = patience_counter + 1;
        end
        if patience_counter >= patience
            fprintf('Early stopping at epoch %d (best epoch %d, best val %.4f)\n', epoch, best_epoch, best_val);
            break;
        end
    end

    student_net = best_net;
    train_info = struct('train_loss', train_loss, 'val_loss', val_loss, ...
        'best_epoch', best_epoch, 'best_val', best_val);
end

function [gradients, loss] = modelGradientsKD(student_net, teacher_dlnet, dlX, dlXecg, dlY, ...
    enable_kd, kd_feature_layer, kd_output_layer, kd_feature_weight, kd_response_weight, kd_use_global_pool)
    [dlYpred, dlFeatS] = forward(student_net, dlX, 'Outputs', {kd_output_layer, kd_feature_layer});
    loss_sup = mse_loss(dlYpred, dlY);

    if enable_kd
        [dlYt, dlFeatT] = forward(teacher_dlnet, dlXecg, 'Outputs', {kd_output_layer, kd_feature_layer});
        if kd_use_global_pool
            featS = global_avg_pool(dlFeatS);
            featT = global_avg_pool(dlFeatT);
            loss_feat = mse_loss(featS, featT);
        else
            loss_feat = mse_loss(dlFeatS, dlFeatT);
        end
        if kd_response_weight > 0
            loss_resp = mse_loss(dlYpred, dlYt);
        else
            loss_resp = dlarray(0.0);
        end
    else
        loss_feat = dlarray(0.0);
        loss_resp = dlarray(0.0);
    end

    loss = loss_sup + kd_feature_weight * loss_feat + kd_response_weight * loss_resp;
    gradients = dlgradient(loss, student_net.Learnables);
end

function loss = evaluate_kd_loss(student_net, teacher_dlnet, Xr4, Xecg4, Y4, mini_batch, ...
    enable_kd, kd_feature_layer, kd_output_layer, kd_feature_weight, kd_response_weight, kd_use_global_pool)
    num_obs = size(Xr4, 4);
    if num_obs == 0
        loss = NaN;
        return;
    end
    num_iter = max(1, ceil(num_obs / mini_batch));
    loss_sum = 0;
    for i = 1:mini_batch:num_obs
        batch = i:min(i+mini_batch-1, num_obs);
        dlX = dlarray(single(Xr4(:, :, :, batch)), 'SSCB');
        dlXecg = dlarray(single(Xecg4(:, :, :, batch)), 'SSCB');
        dlY = dlarray(single(Y4(:, :, :, batch)), 'SSCB');
        loss_batch = modelLossKD(student_net, teacher_dlnet, dlX, dlXecg, dlY, ...
            enable_kd, kd_feature_layer, kd_output_layer, kd_feature_weight, kd_response_weight, kd_use_global_pool);
        loss_sum = loss_sum + double(gather(extractdata(loss_batch)));
    end
    loss = loss_sum / num_iter;
end

function loss = modelLossKD(student_net, teacher_dlnet, dlX, dlXecg, dlY, ...
    enable_kd, kd_feature_layer, kd_output_layer, kd_feature_weight, kd_response_weight, kd_use_global_pool)
    [dlYpred, dlFeatS] = forward(student_net, dlX, 'Outputs', {kd_output_layer, kd_feature_layer});
    loss_sup = mse_loss(dlYpred, dlY);

    if enable_kd
        [dlYt, dlFeatT] = forward(teacher_dlnet, dlXecg, 'Outputs', {kd_output_layer, kd_feature_layer});
        if kd_use_global_pool
            featS = global_avg_pool(dlFeatS);
            featT = global_avg_pool(dlFeatT);
            loss_feat = mse_loss(featS, featT);
        else
            loss_feat = mse_loss(dlFeatS, dlFeatT);
        end
        if kd_response_weight > 0
            loss_resp = mse_loss(dlYpred, dlYt);
        else
            loss_resp = dlarray(0.0);
        end
    else
        loss_feat = dlarray(0.0);
        loss_resp = dlarray(0.0);
    end

    loss = loss_sup + kd_feature_weight * loss_feat + kd_response_weight * loss_resp;
end

function dlnet = prepare_dlnet_from_lgraph(lgraph)
    if any(strcmp({lgraph.Layers.Name}, 'reg'))
        lgraph = removeLayers(lgraph, 'reg');
    end
    dlnet = dlnetwork(lgraph);
end

function dlnet = make_dlnet_from_trained(net)
    if isa(net, 'dlnetwork')
        dlnet = net;
        return;
    end
    try
        lgraph = layerGraph(net);
    catch
        lgraph = layerGraph(net.Layers);
    end
    if any(strcmp({lgraph.Layers.Name}, 'reg'))
        lgraph = removeLayers(lgraph, 'reg');
    end
    dlnet = dlnetwork(lgraph);
end

function loss = mse_loss(a, b)
    diff = a - b;
    loss = mean(diff.^2, 'all');
end

function y = global_avg_pool(x)
    % Global average pool across spatial dims for modality-robust KD
    y = mean(x, [1 2]);
end

function [net, train_info] = train_heart_net(Xtr4, Ytr4, Xva4, Yva4, Xte4, Yte4, win_len, ...
    model_arch, enable_template_init, fs_ds, template_sigma_sec, ...
    enable_dual_head_multitask, enable_temporal_attention)
    if size(Xtr4, 4) < 2
        error('Not enough training windows.');
    end
    if size(Xva4, 4) < 1
        error('Validation set is empty. Add more paired data.');
    end

    n_out = 1 + double(enable_dual_head_multitask);
    switch lower(model_arch)
        case 'unet'
            lgraph = create_unet_lgraph(win_len, fs_ds, enable_template_init, template_sigma_sec, n_out, enable_temporal_attention);
        otherwise
            lgraph = create_resnet_lgraph(win_len, n_out, enable_temporal_attention);
    end

    opts = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 5e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.3, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 5e-5, ...
        'GradientThreshold', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {Xva4, Yva4}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5, ...
        'OutputNetwork', 'best-validation-loss', ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    [net, info] = trainNetwork(Xtr4, Ytr4, lgraph, opts);

    train_info = struct();
    train_info.train_loss = info.TrainingLoss(:);
    if isfield(info, 'ValidationLoss')
        train_info.val_loss = info.ValidationLoss(:);
    else
        train_info.val_loss = [];
    end

    if ~isempty(Xte4)
        Yte_pred = predict(net, Xte4, 'MiniBatchSize', 256);
        Yte_true_wave = Yte4(:, :, 1, :);
        Yte_pred_wave = Yte_pred(:, :, 1, :);
        train_info.test_rmse = sqrt(mean((Yte_true_wave - Yte_pred_wave).^2, 'all'));
        train_info.test_corr = corr(Yte_true_wave(:), Yte_pred_wave(:));
        fprintf('Hold-out test RMSE: %.4f, Corr: %.4f\n', train_info.test_rmse, train_info.test_corr);
    end
end

function lgraph = create_resnet_lgraph(win_len, n_out, enable_temporal_attention)
    layers = [
        imageInputLayer([win_len 1 1], 'Normalization', 'none', 'Name', 'in')
        convolution2dLayer([7 1], 32, 'Padding', 'same', 'Name', 'conv_init')
        batchNormalizationLayer('Name', 'bn_init')
        reluLayer('Name', 'relu_init')

        convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv1a')
        batchNormalizationLayer('Name', 'bn1a')
        reluLayer('Name', 'relu1a')
        convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv1b')
        batchNormalizationLayer('Name', 'bn1b')
        additionLayer(2, 'Name', 'add1')
        reluLayer('Name', 'relu1')

        convolution2dLayer([5 1], 64, 'Padding', 'same', 'Stride', [2 1], 'Name', 'conv2a')
        batchNormalizationLayer('Name', 'bn2a')
        reluLayer('Name', 'relu2a')
        convolution2dLayer([5 1], 64, 'Padding', 'same', 'Name', 'conv2b')
        batchNormalizationLayer('Name', 'bn2b')
        additionLayer(2, 'Name', 'add2')    
        reluLayer('Name', 'relu2')

        convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv3a')
        batchNormalizationLayer('Name', 'bn3a')
        reluLayer('Name', 'relu3a')
        convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv3b')
        batchNormalizationLayer('Name', 'bn3b')
        additionLayer(2, 'Name', 'add3')
        reluLayer('Name', 'relu3')

        dropoutLayer(0.25, 'Name', 'drop')
        transposedConv2dLayer([4 1], 64, 'Stride', [2 1], 'Cropping', 'same', 'Name', 'upconv')
        batchNormalizationLayer('Name', 'bn_up')
        reluLayer('Name', 'relu_up')
        convolution2dLayer([3 1], 16, 'Padding', 'same', 'Name', 'conv_out1')
        reluLayer('Name', 'relu_out')
        convolution2dLayer([1 1], n_out, 'Padding', 'same', 'Name', 'conv_out2')
        regressionLayer('Name', 'reg')
    ];

    lgraph = layerGraph(layers);

    lgraph = addLayers(lgraph, convolution2dLayer([1 1], 32, 'Name', 'skip1'));
    lgraph = connectLayers(lgraph, 'relu_init', 'skip1');
    lgraph = connectLayers(lgraph, 'skip1', 'add1/in2');

    lgraph = addLayers(lgraph, convolution2dLayer([1 1], 64, 'Stride', [2 1], 'Name', 'skip2'));
    lgraph = connectLayers(lgraph, 'relu1', 'skip2');
    lgraph = connectLayers(lgraph, 'skip2', 'add2/in2');

    lgraph = addLayers(lgraph, convolution2dLayer([1 1], 64, 'Name', 'skip3'));
    lgraph = connectLayers(lgraph, 'relu2', 'skip3');
    lgraph = connectLayers(lgraph, 'skip3', 'add3/in2');

    if enable_temporal_attention
        lgraph = add_temporal_attention_block(lgraph, 'relu3', 'attn3', 64);
        lgraph = disconnectLayers(lgraph, 'relu3', 'drop');
        lgraph = connectLayers(lgraph, 'attn3_add', 'drop');
    end
end

function lgraph = create_unet_lgraph(win_len, fs_ds, enable_template_init, template_sigma_sec, n_out, enable_temporal_attention)
    enc1_conv1 = convolution2dLayer([9 1], 32, 'Padding', 'same', 'Name', 'enc1_conv1');
    if enable_template_init
        [w0, b0] = make_gaussian_template_bank(9, 32, fs_ds, template_sigma_sec);
        enc1_conv1.Weights = w0;
        enc1_conv1.Bias = b0;
    end

    layers = [
        imageInputLayer([win_len 1 1], 'Normalization', 'none', 'Name', 'in')
        enc1_conv1
        batchNormalizationLayer('Name', 'enc1_bn1')
        reluLayer('Name', 'enc1_relu1')
        convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'enc1_conv2')
        batchNormalizationLayer('Name', 'enc1_bn2')
        reluLayer('Name', 'enc1_relu2')

        convolution2dLayer([4 1], 64, 'Stride', [2 1], 'Padding', 'same', 'Name', 'down1')
        batchNormalizationLayer('Name', 'enc2_bn0')
        reluLayer('Name', 'enc2_relu0')
        convolution2dLayer([5 1], 64, 'Padding', 'same', 'Name', 'enc2_conv1')
        batchNormalizationLayer('Name', 'enc2_bn1')
        reluLayer('Name', 'enc2_relu1')
        convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'enc2_conv2')
        batchNormalizationLayer('Name', 'enc2_bn2')
        reluLayer('Name', 'enc2_relu2')

        convolution2dLayer([4 1], 128, 'Stride', [2 1], 'Padding', 'same', 'Name', 'down2')
        batchNormalizationLayer('Name', 'bot_bn0')
        reluLayer('Name', 'bot_relu0')
        convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'bot_conv1')
        batchNormalizationLayer('Name', 'bot_bn1')
        reluLayer('Name', 'bot_relu1')
        convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'bot_conv2')
        batchNormalizationLayer('Name', 'bot_bn2')
        reluLayer('Name', 'bot_relu2')

        transposedConv2dLayer([4 1], 64, 'Stride', [2 1], 'Cropping', 'same', 'Name', 'up2')
        depthConcatenationLayer(2, 'Name', 'dec2_concat')
        convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'dec2_conv1')
        batchNormalizationLayer('Name', 'dec2_bn1')
        reluLayer('Name', 'dec2_relu1')
        convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'dec2_conv2')
        batchNormalizationLayer('Name', 'dec2_bn2')
        reluLayer('Name', 'dec2_relu2')

        transposedConv2dLayer([4 1], 32, 'Stride', [2 1], 'Cropping', 'same', 'Name', 'up1')
        depthConcatenationLayer(2, 'Name', 'dec1_concat')
        convolution2dLayer([3 1], 32, 'Padding', 'same', 'Name', 'dec1_conv1')
        batchNormalizationLayer('Name', 'dec1_bn1')
        reluLayer('Name', 'dec1_relu1')
        convolution2dLayer([3 1], 32, 'Padding', 'same', 'Name', 'dec1_conv2')
        batchNormalizationLayer('Name', 'dec1_bn2')
        reluLayer('Name', 'dec1_relu2')

        dropoutLayer(0.25, 'Name', 'drop')
        convolution2dLayer([1 1], n_out, 'Padding', 'same', 'Name', 'conv_out')
        regressionLayer('Name', 'reg')
    ];

    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2');
    lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2');

    if enable_temporal_attention
        lgraph = add_temporal_attention_block(lgraph, 'bot_relu2', 'bot_attn', 128);
        lgraph = disconnectLayers(lgraph, 'bot_relu2', 'up2');
        lgraph = connectLayers(lgraph, 'bot_attn_add', 'up2');
    end
end

function lgraph = add_temporal_attention_block(lgraph, input_name, prefix, n_channels)
    n_mid = max(8, floor(n_channels / 4));
    attn_layers = [
        convolution2dLayer([1 1], n_mid, 'Padding', 'same', 'Name', [prefix '_reduce'])
        reluLayer('Name', [prefix '_relu'])
        convolution2dLayer([7 1], n_channels, 'Padding', 'same', 'Name', [prefix '_expand'])
        sigmoidLayer('Name', [prefix '_sig'])
        multiplicationLayer(2, 'Name', [prefix '_mul'])
        additionLayer(2, 'Name', [prefix '_add'])
    ];
    lgraph = addLayers(lgraph, attn_layers);
    lgraph = connectLayers(lgraph, input_name, [prefix '_reduce']);
    lgraph = connectLayers(lgraph, input_name, [prefix '_mul/in2']);
    lgraph = connectLayers(lgraph, input_name, [prefix '_add/in2']);
end

function [w, b] = make_gaussian_template_bank(filter_len, n_filters, fs_ds, sigma_sec)
    sigma = max(0.5, sigma_sec * fs_ds);
    t = (1:filter_len) - (filter_len + 1) / 2;
    g = exp(-0.5 * (t / sigma).^2);
    g = g - mean(g);
    g = g / (norm(g) + eps);

    w = zeros(filter_len, 1, 1, n_filters, 'single');
    for k = 1:n_filters
        shift = round(((k - 1) / max(1, n_filters - 1)) * (filter_len - 1)) - floor((filter_len - 1) / 2);
        w(:, 1, 1, k) = single(circshift(g(:), shift));
    end
    b = zeros(1, 1, n_filters, 'single');
end

function Yg = build_peak_guided_targets(Y, fs_ds, valid_hr_bpm, sigma_sec, alpha)
    if isempty(Y)
        Yg = Y;
        return;
    end
    alpha = max(0, min(0.8, alpha));
    win_len = size(Y, 1);
    n = size(Y, 2);
    Yg = zeros(size(Y), 'like', Y);

    sigma = max(1, round(sigma_sec * fs_ds));
    k = (-round(0.20 * fs_ds):round(0.20 * fs_ds))';
    g = exp(-0.5 * (k / sigma).^2);
    g = g / (sum(g) + eps);
    min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs_ds));

    for i = 1:n
        y = double(Y(:, i));
        y = y - mean(y);
        y = y / (std(y) + eps);
        ymad = mad(y, 1);
        ystd = std(y);
        pmin = max(0.15 * ystd, 1.2 * ymad);
        [~, locs] = findpeaks(y, 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', pmin);
        if numel(locs) < 2
            pmin2 = max(0.10 * ystd, 0.8 * ymad);
            [~, locs] = findpeaks(abs(y), 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', pmin2);
        end

        pulse = zeros(win_len, 1);
        if ~isempty(locs)
            pulse(locs) = 1;
        end
        y_peak = conv(pulse, g, 'same');
        y_peak = y_peak - mean(y_peak);
        y_peak = y_peak / (std(y_peak) + eps);

        y_mix = (1 - alpha) * y + alpha * y_peak;
        y_mix = y_mix - mean(y_mix);
        y_mix = y_mix / (std(y_mix) + eps);
        Yg(:, i) = cast(y_mix, 'like', Y);
    end
end

function Yp = build_peak_map_targets(Y, fs_ds, valid_hr_bpm, sigma_sec)
    if isempty(Y)
        Yp = Y;
        return;
    end
    win_len = size(Y, 1);
    n = size(Y, 2);
    Yp = zeros(size(Y), 'like', Y);

    sigma = max(1, round(sigma_sec * fs_ds));
    k = (-round(0.20 * fs_ds):round(0.20 * fs_ds))';
    g = exp(-0.5 * (k / sigma).^2);
    g = g / (sum(g) + eps);
    min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs_ds));

    for i = 1:n
        y = double(Y(:, i));
        y = y - mean(y);
        y = y / (std(y) + eps);
        ymad = mad(y, 1);
        ystd = std(y);
        pmin = max(0.15 * ystd, 1.2 * ymad);
        [~, locs] = findpeaks(y, 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', pmin);
        if numel(locs) < 2
            pmin2 = max(0.10 * ystd, 0.8 * ymad);
            [~, locs] = findpeaks(abs(y), 'MinPeakDistance', min_peak_dist, 'MinPeakProminence', pmin2);
        end

        pulse = zeros(win_len, 1);
        if ~isempty(locs)
            pulse(locs) = 1;
        end
        y_peak = conv(pulse, g, 'same');
        y_peak = y_peak - mean(y_peak);
        y_peak = y_peak / (std(y_peak) + eps);
        Yp(:, i) = cast(y_peak, 'like', Y);
    end
end

function [Xaug, Xecg_aug, Yaug] = augment_data_kd(X, Xecg, Y, enable_advanced_time_augmentation)
    n = size(X, 2);
    if n == 0
        Xaug = X;
        Xecg_aug = Xecg;
        Yaug = Y;
        return;
    end
    if nargin < 4
        enable_advanced_time_augmentation = false;
    end

    n_noise = round(n * 0.30);
    n_scale = round(n * 0.25);
    if enable_advanced_time_augmentation
        n_shift = round(n * 0.10);
        n_mask = round(n * 0.05);
        n_warp = round(n * 0.08);
    else
        n_shift = 0;
        n_mask = 0;
        n_warp = 0;
    end
    max_aug = n + n_noise + n_scale + n_shift + n_mask + n_warp;

    Xaug = zeros(size(X, 1), max_aug, 'like', X);
    Xecg_aug = zeros(size(Xecg, 1), max_aug, 'like', Xecg);
    Yaug = zeros(size(Y, 1), max_aug, 'like', Y);
    Xaug(:, 1:n) = X;
    Xecg_aug(:, 1:n) = Xecg;
    Yaug(:, 1:n) = Y;
    idx = n;

    if n_noise > 0
        noise_idx = randperm(n, n_noise);
        noise = randn(size(X, 1), n_noise) * 0.03;
        Xaug(:, idx+1:idx+n_noise) = X(:, noise_idx) + noise;
        Xecg_aug(:, idx+1:idx+n_noise) = Xecg(:, noise_idx);
        Yaug(:, idx+1:idx+n_noise) = Y(:, noise_idx);
        idx = idx + n_noise;
    end

    if n_scale > 0
        scale_idx = randperm(n, n_scale);
        scales = 0.85 + 0.30 * rand(1, n_scale);
        Xaug(:, idx+1:idx+n_scale) = X(:, scale_idx) .* scales;
        Xecg_aug(:, idx+1:idx+n_scale) = Xecg(:, scale_idx) .* scales;
        Yaug(:, idx+1:idx+n_scale) = Y(:, scale_idx) .* scales;
        idx = idx + n_scale;
    end

    if n_shift > 0
        shift_idx = randperm(n, n_shift);
        max_shift = max(1, round(0.08 * size(X, 1)));
        for k = 1:n_shift
            sh = randi([-max_shift, max_shift]);
            idx = idx + 1;
            Xaug(:, idx) = circshift(X(:, shift_idx(k)), sh);
            Xecg_aug(:, idx) = circshift(Xecg(:, shift_idx(k)), sh);
            Yaug(:, idx) = circshift(Y(:, shift_idx(k)), sh);
        end
    end

    if n_mask > 0
        mask_idx = randperm(n, n_mask);
        L = size(X, 1);
        for k = 1:n_mask
            idx = idx + 1;
            xk = X(:, mask_idx(k));
            ek = Xecg(:, mask_idx(k));
            yk = Y(:, mask_idx(k));
            w = randi([max(3, round(0.05 * L)), max(4, round(0.15 * L))]);
            s = randi([1, L - w + 1]);
            xk(s:s+w-1) = 0.0;
            ek(s:s+w-1) = 0.0;
            Xaug(:, idx) = xk;
            Xecg_aug(:, idx) = ek;
            Yaug(:, idx) = yk;
        end
    end

    if n_warp > 0
        warp_idx = randperm(n, n_warp);
        L = size(X, 1);
        t0 = linspace(0, 1, L)';
        for k = 1:n_warp
            idx = idx + 1;
            xk = X(:, warp_idx(k));
            ek = Xecg(:, warp_idx(k));
            yk = Y(:, warp_idx(k));
            anchors = [0; 0.33; 0.66; 1];
            jitter = [0; (rand(2,1) - 0.5) * 0.24; 0];
            t_map = anchors + jitter;
            t_map = min(max(t_map, 0), 1);
            t_map = cummax(t_map);
            t_map(end) = 1;
            tw = interp1(anchors, t_map, t0, 'pchip');
            tw = (tw - tw(1)) / (tw(end) - tw(1) + eps);
            Xaug(:, idx) = interp1(t0, xk, tw, 'linear', 'extrap');
            Xecg_aug(:, idx) = interp1(t0, ek, tw, 'linear', 'extrap');
            Yaug(:, idx) = interp1(t0, yk, tw, 'linear', 'extrap');
        end
    end

    Xaug = Xaug(:, 1:idx);
    Xecg_aug = Xecg_aug(:, 1:idx);
    Yaug = Yaug(:, 1:idx);
end

function [Xaug, Yaug] = augment_data(X, Y, enable_advanced_time_augmentation)
    n = size(X, 2);
    if n == 0
        Xaug = X;
        Yaug = Y;
        return;
    end
    if nargin < 3
        enable_advanced_time_augmentation = false;
    end

    n_noise = round(n * 0.30);
    n_scale = round(n * 0.25);
    if enable_advanced_time_augmentation
        n_shift = round(n * 0.10);
        n_mask = round(n * 0.05);
        n_warp = round(n * 0.08);
    else
        n_shift = 0;
        n_mask = 0;
        n_warp = 0;
    end
    max_aug = n + n_noise + n_scale + n_shift + n_mask + n_warp;

    Xaug = zeros(size(X, 1), max_aug, 'like', X);
    Yaug = zeros(size(Y, 1), max_aug, 'like', Y);
    Xaug(:, 1:n) = X;
    Yaug(:, 1:n) = Y;
    idx = n;

    if n_noise > 0
        noise_idx = randperm(n, n_noise);
        noise = randn(size(X, 1), n_noise) * 0.03;
        Xaug(:, idx+1:idx+n_noise) = X(:, noise_idx) + noise;
        Yaug(:, idx+1:idx+n_noise) = Y(:, noise_idx);
        idx = idx + n_noise;
    end

    if n_scale > 0
        scale_idx = randperm(n, n_scale);
        scales = 0.85 + 0.30 * rand(1, n_scale);
        Xaug(:, idx+1:idx+n_scale) = X(:, scale_idx) .* scales;
        Yaug(:, idx+1:idx+n_scale) = Y(:, scale_idx) .* scales;
        idx = idx + n_scale;
    end

    if n_shift > 0
        shift_idx = randperm(n, n_shift);
        max_shift = max(1, round(0.08 * size(X, 1)));
        for k = 1:n_shift
            sh = randi([-max_shift, max_shift]);
            idx = idx + 1;
            Xaug(:, idx) = circshift(X(:, shift_idx(k)), sh);
            Yaug(:, idx) = circshift(Y(:, shift_idx(k)), sh);
        end
    end

    if n_mask > 0
        mask_idx = randperm(n, n_mask);
        L = size(X, 1);
        for k = 1:n_mask
            idx = idx + 1;
            xk = X(:, mask_idx(k));
            yk = Y(:, mask_idx(k));
            w = randi([max(3, round(0.05 * L)), max(4, round(0.15 * L))]);
            s = randi([1, L - w + 1]);
            xk(s:s+w-1) = 0.0;
            Xaug(:, idx) = xk;
            Yaug(:, idx) = yk;
        end
    end

    if n_warp > 0
        warp_idx = randperm(n, n_warp);
        L = size(X, 1);
        t0 = linspace(0, 1, L)';
        for k = 1:n_warp
            idx = idx + 1;
            xk = X(:, warp_idx(k));
            yk = Y(:, warp_idx(k));
            anchors = [0; 0.33; 0.66; 1];
            jitter = [0; (rand(2,1) - 0.5) * 0.24; 0];
            t_map = anchors + jitter;
            t_map = min(max(t_map, 0), 1);
            t_map = cummax(t_map);
            t_map(end) = 1;
            tw = interp1(anchors, t_map, t0, 'pchip');
            tw = (tw - tw(1)) / (tw(end) - tw(1) + eps);
            Xaug(:, idx) = interp1(t0, xk, tw, 'linear', 'extrap');
            Yaug(:, idx) = interp1(t0, yk, tw, 'linear', 'extrap');
        end
    end

    Xaug = Xaug(:, 1:idx);
    Yaug = Yaug(:, 1:idx);
end

function metrics = evaluate_performance(y_pred, y_true, Y_true_win, Y_pred_win, fs, heart_band_hz)
    y_pred = y_pred(:);
    y_true = y_true(:);
    min_len = min(numel(y_pred), numel(y_true));
    y_pred = y_pred(1:min_len);
    y_true = y_true(1:min_len);

    metrics.mae = mean(abs(y_pred - y_true));
    metrics.rmse = sqrt(mean((y_pred - y_true).^2));
    signal_power = mean(y_true.^2);
    noise_power = mean((y_pred - y_true).^2);
    metrics.snr = 10 * log10(signal_power / (noise_power + eps));
    metrics.corr = corr(y_pred, y_true);

    Y_true_vec = Y_true_win(:);
    Y_pred_vec = Y_pred_win(:);
    metrics.window_mae = mean(abs(Y_pred_vec - Y_true_vec));
    metrics.window_rmse = sqrt(mean((Y_pred_vec - Y_true_vec).^2));

    N = min(numel(y_pred), numel(y_true));
    nfft = 2^nextpow2(max(N, 16));
    Yp = abs(fft(y_pred(1:N), nfft));
    Yt = abs(fft(y_true(1:N), nfft));
    f = (0:nfft-1)' * fs / nfft;
    half = 1:(floor(nfft/2)+1);
    Yp = Yp(half);
    Yt = Yt(half);
    f = f(half);
    metrics.spectral_rmse = sqrt(mean((Yp - Yt).^2));
    band = f >= heart_band_hz(1) & f <= heart_band_hz(2);
    e_pred = sum(Yp(band).^2);
    e_true = sum(Yt(band).^2);
    metrics.band_energy_relerr = abs(e_pred - e_true) / (e_true + eps);
end

function [hr_inst, t_hr, pks, locs] = estimate_hr_from_signal(sig, fs, valid_hr_bpm, use_adaptive_threshold)
    sig = sig(:);
    min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs));
    if nargin < 4
        use_adaptive_threshold = true;
    end

    signal_std = std(sig);
    signal_mad = mad(sig, 1);
    if use_adaptive_threshold
        sqi = compute_signal_quality_index(sig, fs, valid_hr_bpm);
        if sqi > 0.8
            prom_factor = 0.15;
            height_factor = 0.20;
        elseif sqi > 0.5
            prom_factor = 0.25;
            height_factor = 0.30;
        else
            prom_factor = 0.35;
            height_factor = 0.40;
        end
    else
        prom_factor = 0.15;
        height_factor = 0.20;
    end

    peak_thresh = max(0.12 * std(sig), 1.0 * mad(sig));
    prom_thresh = max(0.10 * std(sig), 1.0 * mad(sig));

    %%peak_thresh = max(height_factor * signal_std, 1.5 * signal_mad);
    %%prom_thresh = max(prom_factor * signal_std, 1.2 * signal_mad);

    [pks, locs] = findpeaks(sig, ...
        'MinPeakDistance', min_peak_dist, ...
        'MinPeakHeight', peak_thresh, ...
        'MinPeakProminence', prom_thresh);

    if numel(locs) < 3
        [pks_n, locs_n] = findpeaks(-sig, ...
            'MinPeakDistance', min_peak_dist, ...
            'MinPeakHeight', peak_thresh, ...
            'MinPeakProminence', prom_thresh);
        if numel(locs_n) > numel(locs)
            pks = -pks_n;
            locs = locs_n;
        end
    end

    if numel(locs) < 3 && use_adaptive_threshold
        peak_thresh2 = 0.75 * peak_thresh;
        prom_thresh2 = 0.75 * prom_thresh;
        [pks, locs] = findpeaks(sig, ...
            'MinPeakDistance', min_peak_dist, ...
            'MinPeakHeight', peak_thresh2, ...
            'MinPeakProminence', prom_thresh2);
    end

    hr_inst = [];
    t_hr = [];
    if numel(locs) < 3
        return;
    end

    rr = diff(locs) / fs;
    hr_all = 60 ./ rr;
    valid = hr_all >= valid_hr_bpm(1) & hr_all <= valid_hr_bpm(2);
    hr_inst = hr_all(valid);
    t_all = (locs(2:end)-1) / fs;
    t_hr = t_all(valid);
end

function sqi = compute_signal_quality_index(sig, fs, valid_hr_bpm)
    sig = sig(:);
    N = numel(sig);
    if N < 16 || fs <= 0
        sqi = 0;
        return;
    end
    Y = abs(fft(sig));
    freqs = (0:N-1)' * fs / N;
    heart_band = [valid_hr_bpm(1), valid_hr_bpm(2)] / 60;
    idx_band = freqs >= heart_band(1) & freqs <= heart_band(2);
    e_band = sum(Y(idx_band).^2);
    e_all = sum(Y.^2);
    energy_ratio = e_band / (e_all + eps);

    min_peak_dist = max(1, round((60 / valid_hr_bpm(2)) * fs));
    [~, locs] = findpeaks(sig, 'MinPeakDistance', min_peak_dist);
    if numel(locs) > 2
        rr = diff(locs);
        cv = std(rr) / (mean(rr) + eps);
        regularity = exp(-cv);
    else
        regularity = 0;
    end
    sqi = max(0, min(1, 0.6 * energy_ratio + 0.4 * regularity));
end

function hr_smooth = kalman_smooth_hr(hr_inst, t_hr)
    hr_inst = hr_inst(:);
    if nargin < 2
        t_hr = (0:numel(hr_inst)-1)';
    else
        t_hr = t_hr(:);
    end
    n = numel(hr_inst);
    if n < 3
        hr_smooth = hr_inst;
        return;
    end

    dt = median(diff(t_hr));
    if ~isfinite(dt) || dt <= 0
        dt = 1;
    end
    A = [1, dt; 0, 1];
    H = [1, 0];
    dhr = diff(hr_inst);
    dhr = dhr(isfinite(dhr));
    if isempty(dhr)
        q_vel = 0.05;
    else
        q_vel = max(0.05, 0.2 * std(dhr));
    end
    Q = [0.05, 0; 0, q_vel];
    hr_valid = hr_inst(isfinite(hr_inst));
    if isempty(hr_valid)
        R = 1.0;
    else
        R = max(1.0, 0.5 * var(hr_valid));
    end

    x = [hr_inst(1); 0];
    P = eye(2);
    hr_smooth = zeros(n, 1);
    for i = 1:n
        x = A * x;
        P = A * P * A' + Q;
        K = P * H' / (H * P * H' + R);
        x = x + K * (hr_inst(i) - H * x);
        P = (eye(2) - K * H) * P;
        hr_smooth(i) = x(1);
    end
end

function [mae_bpm, rmse_bpm] = compare_hr_series(t1, hr1, t2, hr2)
    if isempty(t1) || isempty(hr1) || isempty(t2) || isempty(hr2)
        mae_bpm = NaN;
        rmse_bpm = NaN;
        return;
    end
    t_min = max(min(t1), min(t2));
    t_max = min(max(t1), max(t2));
    if t_max <= t_min
        mae_bpm = NaN;
        rmse_bpm = NaN;
        return;
    end
    t_grid = linspace(t_min, t_max, 200);
    hr1i = interp1(t1, hr1, t_grid, 'linear', 'extrap');
    hr2i = interp1(t2, hr2, t_grid, 'linear', 'extrap');
    err = hr1i - hr2i;
    mae_bpm = mean(abs(err));
    rmse_bpm = sqrt(mean(err.^2));
end

function [W, starts] = make_windows(x, win_len, hop_len)
    x = x(:);
    L = numel(x);
    starts = 1:hop_len:(L - win_len + 1);
    if isempty(starts)
        starts = 1;
        x = [x; zeros(win_len - L, 1)];
    end

    W = zeros(win_len, numel(starts), 'single');
    for i = 1:numel(starts)
        seg = x(starts(i):starts(i)+win_len-1);
        seg = seg - mean(seg);
        seg = seg / (std(seg) + eps);
        W(:, i) = single(seg);
    end
end

function y = overlap_add(W, starts, out_len)
    win_len = size(W, 1);
    y = zeros(out_len, 1);
    wsum = zeros(out_len, 1);
    
    % 计算重叠率并选择合适的窗口
    if numel(starts) > 1
        hop_len = starts(2) - starts(1);
        overlap_ratio = 1 - hop_len / win_len;
    else
        overlap_ratio = 0.75;
    end
    
    % 根据重叠率选择窗口函数
    if overlap_ratio > 0.5
        % 高重叠率：使用平方根Hann窗，避免边缘过度衰减
        w = sqrt(hann(win_len, 'periodic'));
    else
        % 低重叠率：使用标准Hann
        w = hann(win_len, 'periodic');
    end

    for i = 1:numel(starts)
        idx = starts(i):(starts(i)+win_len-1);
        idx = idx(idx <= out_len);
        k = 1:numel(idx);
        y(idx) = y(idx) + W(k, i) .* w(k);
        wsum(idx) = wsum(idx) + w(k);
    end

    % 避免除以过小的权重值，防止数值不稳定
    min_weight = 0.01;
    wsum(wsum < min_weight) = 1.0;
    y = y ./ wsum;
end

function x = robust_clip(x, k)
    x = x(:);
    med = median(x, 'omitnan');
    madv = mad(x, 1);
    if ~isfinite(madv) || madv <= 0
        return;
    end
    lim = k * madv;
    x = min(max(x, med - lim), med + lim);
end
