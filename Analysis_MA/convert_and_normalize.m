%% convert_and_normalize.m
% Converts per-condition .mat feature files to normalized CSV files
% ready for the DLinear model (run_linear_v1.py / run_all_conditions.py).
%
% For each condition file (e.g. SFC_S_features.mat):
%   1. Loads the features matrix [N x 66]
%   2. Normalizes each column to [-1, 1] using min-max scaling:
%        x_norm = 2 * (x - min) / (max - min) - 1
%   3. Creates a synthetic date column starting at 2016-07-01 00:00:00
%      with 12.5 Hz sampling (matching DATA_MA/convert_mat_to_csv.m)
%   4. Saves as CSV with columns: date, F1, F2, ..., F66
%
% Input:  Analysis_MA/data/*_features.mat
% Output: Analysis_MA/data/*_normalized.csv
%
% This mirrors the pipeline in:
%   DATA_MA/convert_mat_to_csv.m  (mat -> csv with date)
%   data_normalization.py          (min-max to [-1,1])

%% INIT ===================================================================
clc; clear; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));  % LTSF-Linear root
data_dir = fullfile(repo_root, 'Analysis_MA', 'data');

% List all *_features.mat files
mat_files = dir(fullfile(data_dir, '*_features.mat'));

if isempty(mat_files)
    error('No *_features.mat files found in %s. Run extract_cohort_data.m first.', data_dir);
end

fprintf('Found %d feature files to convert.\n\n', length(mat_files));

%% Process each file ======================================================
for i = 1:length(mat_files)
    mat_path = fullfile(data_dir, mat_files(i).name);
    fprintf('Processing: %s\n', mat_files(i).name);

    % Load features [N x 66]
    S = load(mat_path, 'features');
    features = S.features;
    [n_frames, n_feat] = size(features);
    fprintf('  Shape: %d frames x %d features\n', n_frames, n_feat);

    % --- Normalize each column to [-1, 1] --------------------------------
    features_norm = zeros(size(features));
    for col = 1:n_feat
        col_min = min(features(:, col));
        col_max = max(features(:, col));
        col_range = col_max - col_min;

        if col_range == 0
            % Constant column -> set to 0
            features_norm(:, col) = 0;
            fprintf('  WARNING: Column F%d is constant (value=%.4f), set to 0.\n', col, col_min);
        else
            features_norm(:, col) = 2 * (features(:, col) - col_min) / col_range - 1;
        end
    end

    % Verify normalization
    fprintf('  Normalized range: [%.3f, %.3f] (expected [-1, 1])\n', ...
        min(features_norm(:)), max(features_norm(:)));

    % --- Create date column (12.5 Hz, matching DATA_MA convention) -------
    seconds_vec = (0:n_frames-1)' / 12.5;
    base_date = datetime('2016-07-01 00:00:00', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    date_vec = base_date + seconds(seconds_vec);
    date_vec.Format = 'yyyy-MM-dd HH:mm:ss.SSS';

    % --- Build table and save CSV ----------------------------------------
    feat_names = strcat('F', string(1:n_feat));
    T = array2table(features_norm, 'VariableNames', feat_names);
    T = addvars(T, date_vec, 'Before', 1, 'NewVariableNames', 'date');

    % Output filename: replace _features.mat with _normalized.csv
    [~, base_name] = fileparts(mat_files(i).name);
    csv_name = strrep(base_name, '_features', '_normalized');
    csv_path = fullfile(data_dir, [csv_name, '.csv']);

    writetable(T, csv_path);
    fprintf('  Saved: %s\n\n', csv_path);
end

fprintf('=== Conversion and normalization complete ===\n');
fprintf('CSV files are in: %s\n', data_dir);
