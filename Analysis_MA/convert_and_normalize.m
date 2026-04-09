%% convert_and_normalize.m
% Converts per-condition .mat feature files to normalized CSV files
% ready for the DLinear model (run_all_conditions.py).
%
% PART 1 — Per-condition normalization (all 66 features)
%   For each condition file (e.g. SFC_S_features.mat):
%     1. Loads the features matrix [N x 66]
%     2. Normalizes each column to [-1, 1] (min-max per condition)
%     3. Adds a synthetic date column (12.5 Hz)
%     4. Saves as CSV: date, F1, F2, ..., F66
%
% PART 2 — Pooled PCA per cohort
%   For each cohort (SFC, HNG):
%     1. Loads ALL conditions in that cohort
%     2. Concatenates into a single pooled matrix
%     3. Normalizes the pooled data to [-1, 1] (shared min/max)
%     4. Fits PCA on the pooled normalized data
%     5. Selects k components to explain >= 80% of total variance
%     6. Projects each condition onto those shared PCA axes
%     7. Normalizes PCA scores per condition to [-1, 1]
%     8. Saves per-condition CSVs: date, PC1, PC2, ..., PCk
%   Also saves PCA metadata (coefficients, explained variance, k)
%   to <cohort>_pca_info.mat for reference.
%
% This ensures PCA axes are identical within a cohort, making
% cross-condition comparison of prediction errors scientifically valid.
%
% Input:  Analysis_MA/data/*_features.mat
% Output: Analysis_MA/data/*_normalized.csv      (all 66 features)
%         Analysis_MA/data/*_normalized_pca.csv   (pooled PCA features)
%         Analysis_MA/data/<cohort>_pca_info.mat  (PCA metadata)

%% INIT ===================================================================
clc; clear; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));  % LTSF-Linear root
data_dir = fullfile(repo_root, 'Analysis_MA', 'data');

% PCA settings
pca_variance_threshold = 80;  % percent variance to retain

% Define cohorts (must match extract_cohort_data.m)
cohort_defs(1).name       = 'SFC';
cohort_defs(1).conditions = {'S', 'F'};
cohort_defs(2).name       = 'HNG';
cohort_defs(2).conditions = {'H', 'N'};

% Verify feature files exist
mat_files = dir(fullfile(data_dir, '*_features.mat'));
if isempty(mat_files)
    error('No *_features.mat files found in %s. Run extract_cohort_data.m first.', data_dir);
end
fprintf('Found %d feature files.\n\n', length(mat_files));

%% ========================================================================
%  PART 1: Per-condition normalization → *_normalized.csv
%% ========================================================================
fprintf('=== Part 1: Per-condition normalization (all features) ===\n\n');

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
            features_norm(:, col) = 0;
            fprintf('  WARNING: Column F%d is constant (value=%.4f), set to 0.\n', col, col_min);
        else
            features_norm(:, col) = 2 * (features(:, col) - col_min) / col_range - 1;
        end
    end

    fprintf('  Normalized range: [%.3f, %.3f] (expected [-1, 1])\n', ...
        min(features_norm(:)), max(features_norm(:)));

    % --- Date column (12.5 Hz) -------------------------------------------
    seconds_vec = (0:n_frames-1)' / 12.5;
    base_date = datetime('2016-07-01 00:00:00', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    date_vec = base_date + seconds(seconds_vec);
    date_vec.Format = 'yyyy-MM-dd HH:mm:ss.SSS';

    % --- Save CSV --------------------------------------------------------
    feat_names = strcat('F', string(1:n_feat));
    T = array2table(features_norm, 'VariableNames', feat_names);
    T = addvars(T, date_vec, 'Before', 1, 'NewVariableNames', 'date');

    [~, base_name] = fileparts(mat_files(i).name);
    csv_name = strrep(base_name, '_features', '_normalized');
    csv_path = fullfile(data_dir, [csv_name, '.csv']);

    writetable(T, csv_path);
    fprintf('  Saved: %s\n\n', csv_path);
end

%% ========================================================================
%  PART 2: Pooled PCA per cohort → *_normalized_pca.csv
%% ========================================================================
fprintf('=== Part 2: Pooled PCA per cohort (%.0f%% variance threshold) ===\n\n', ...
    pca_variance_threshold);

for ch = 1:length(cohort_defs)
    cohort_name = cohort_defs(ch).name;
    conds       = cohort_defs(ch).conditions;
    num_conds   = length(conds);

    fprintf('--- Cohort: %s (%d conditions) ---\n', cohort_name, num_conds);

    % --- Load all conditions ---------------------------------------------
    raw_data        = cell(1, num_conds);
    n_frames_cond   = zeros(1, num_conds);

    for c = 1:num_conds
        tag = sprintf('%s_%s', cohort_name, conds{c});
        mat_path = fullfile(data_dir, [tag, '_features.mat']);
        if ~exist(mat_path, 'file')
            error('File not found: %s', mat_path);
        end
        S = load(mat_path, 'features');
        raw_data{c} = S.features;
        n_frames_cond(c) = size(S.features, 1);
        fprintf('  Loaded %s: %d frames\n', tag, n_frames_cond(c));
    end

    % --- Pool all conditions ---------------------------------------------
    pooled = vertcat(raw_data{:});
    [n_total, n_feat] = size(pooled);
    fprintf('  Pooled: %d frames x %d features\n', n_total, n_feat);

    % --- Normalize pooled data to [-1, 1] (shared min/max) ---------------
    pooled_norm = zeros(size(pooled));
    col_min   = min(pooled, [], 1);
    col_max   = max(pooled, [], 1);
    col_range = col_max - col_min;

    for col = 1:n_feat
        if col_range(col) == 0
            pooled_norm(:, col) = 0;
            fprintf('  WARNING: Pooled column F%d is constant, set to 0.\n', col);
        else
            pooled_norm(:, col) = 2 * (pooled(:, col) - col_min(col)) / col_range(col) - 1;
        end
    end

    % --- Fit PCA on pooled data ------------------------------------------
    [coeff, score, latent, ~, explained] = pca(pooled_norm);

    cumvar = cumsum(explained);
    n_pca = find(cumvar >= pca_variance_threshold, 1, 'first');

    fprintf('  PCA: %d / %d components explain %.1f%% variance (threshold: %.0f%%)\n', ...
        n_pca, n_feat, cumvar(n_pca), pca_variance_threshold);

    % Print first few components for reference
    n_show = min(n_pca + 3, n_feat);
    fprintf('  Variance explained per PC:\n');
    for pc = 1:n_show
        marker = '';
        if pc == n_pca, marker = '  <-- cutoff'; end
        fprintf('    PC%-3d: %6.2f%%  (cumulative: %6.2f%%)%s\n', ...
            pc, explained(pc), cumvar(pc), marker);
    end

    % --- Save PCA metadata -----------------------------------------------
    pca_info_path = fullfile(data_dir, sprintf('%s_pca_info.mat', cohort_name));
    pca_info = struct();
    pca_info.coeff              = coeff(:, 1:n_pca);
    pca_info.explained          = explained;
    pca_info.cumulative_var     = cumvar;
    pca_info.n_components       = n_pca;
    pca_info.variance_threshold = pca_variance_threshold;
    pca_info.col_min            = col_min;  % pooled normalization params
    pca_info.col_max            = col_max;
    save(pca_info_path, '-struct', 'pca_info', '-v7.3');
    fprintf('  Saved PCA metadata: %s\n', pca_info_path);

    % --- Split scores by condition, normalize, and save ------------------
    start_idx = 1;
    for c = 1:num_conds
        tag  = sprintf('%s_%s', cohort_name, conds{c});
        n_fr = n_frames_cond(c);

        % Extract this condition's PCA scores (shared axes)
        pca_scores = score(start_idx:start_idx + n_fr - 1, 1:n_pca);
        start_idx  = start_idx + n_fr;

        % Normalize PCA scores to [-1, 1] per condition
        % (the model is trained per condition; this keeps input ranges fair)
        for col = 1:n_pca
            cmin = min(pca_scores(:, col));
            cmax = max(pca_scores(:, col));
            cr   = cmax - cmin;
            if cr == 0
                pca_scores(:, col) = 0;
            else
                pca_scores(:, col) = 2 * (pca_scores(:, col) - cmin) / cr - 1;
            end
        end

        % Date column
        seconds_vec = (0:n_fr-1)' / 12.5;
        base_date = datetime('2016-07-01 00:00:00', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        date_vec  = base_date + seconds(seconds_vec);
        date_vec.Format = 'yyyy-MM-dd HH:mm:ss.SSS';

        % Build table and save
        pc_names = strcat('PC', string(1:n_pca));
        T_pca = array2table(pca_scores, 'VariableNames', pc_names);
        T_pca = addvars(T_pca, date_vec, 'Before', 1, 'NewVariableNames', 'date');

        csv_name = sprintf('%s_normalized_pca.csv', tag);
        csv_path = fullfile(data_dir, csv_name);
        writetable(T_pca, csv_path);
        fprintf('  Saved: %s (%d frames x %d PCs)\n', csv_name, n_fr, n_pca);
    end

    fprintf('\n');
end

%% Summary ================================================================
fprintf('=== Conversion and normalization complete ===\n');
fprintf('Output directory: %s\n', data_dir);
fprintf('  *_normalized.csv      -> all 66 features (per-condition normalization)\n');
fprintf('  *_normalized_pca.csv  -> pooled PCA (%.0f%% variance, shared axes per cohort)\n', ...
    pca_variance_threshold);
fprintf('  *_pca_info.mat        -> PCA coefficients and metadata per cohort\n');
