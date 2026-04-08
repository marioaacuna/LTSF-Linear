%% extract_cohort_data.m
% Extracts 66 pose features per condition from each cohort (SFC, HNG).
%
% For each cohort this script:
%   1. Loads raw_concat_analysis.mat  -> analysisstruct
%      (contains jt_features [N x 60] and extra_jt_features [N x ?])
%   2. Loads agg_predictions.mat      -> animal_condition_identifier
%   3. Builds the 66-feature matrix:
%        D = [jt_features, extra_jt_features(:, end-5:end)]
%   4. Maps animal_condition_identifier to per-frame labels via
%      upsampling (repfactor=3) and good-tracking mask.
%   5. Splits frames by condition letter (last char of identifier):
%        SFC -> S (Sham) and F (Formalin)
%        HNG -> H (Hargreaves) and N (Neuropathic)
%   6. Saves one .mat file per condition in Analysis_MA/data/:
%        SFC_S_features.mat, SFC_F_features.mat,
%        HNG_H_features.mat, HNG_N_features.mat
%
% Prerequisites:
%   - Run fbi() or make sure Figures_BEACON_paper is on the MATLAB path
%     so that general_configs() is available.
%   - path_config.json must point to the CAPTURE project root.

%% INIT ===================================================================
clc; clear; close all;

% Add Figures_BEACON_paper to path so we can call general_configs / fbi
repo_root = fileparts(fileparts(mfilename('fullpath')));  % LTSF-Linear root
figures_repo = fullfile(fileparts(repo_root), 'Figures_BEACON_paper');
addpath(genpath(figures_repo));

% Load configuration (paths, parameters)
GC = general_configs();
repfactor = GC.repfactor;  % upsampling factor (typically 3)

% Output directory
output_dir = fullfile(repo_root, 'Analysis_MA', 'data');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% Define cohorts =========================================================
cohorts = struct();

cohorts(1).name            = 'SFC';
cohorts(1).data_folder     = GC.preprocessing_SFC_path;
cohorts(1).conditions      = {'S', 'F'};  % Sham, Formalin
cohorts(1).condition_names = {'Sham', 'Formalin'};

cohorts(2).name            = 'HNG';
cohorts(2).data_folder     = GC.preprocessing_HNG_path;
cohorts(2).conditions      = {'H', 'N'};  % Hargreaves, Neuropathic
cohorts(2).condition_names = {'Hargreaves', 'Neuropathic'};

%% Process each cohort ====================================================
for c = 1:length(cohorts)
    cohort = cohorts(c);
    fprintf('\n=== Processing cohort: %s ===\n', cohort.name);

    % --- Load data -------------------------------------------------------
    file_analysis    = fullfile(cohort.data_folder, 'raw_concat_analysis.mat');
    file_predictions = fullfile(cohort.data_folder, 'agg_predictions.mat');

    if ~exist(file_analysis, 'file')
        error('Analysis file not found: %s', file_analysis);
    end
    if ~exist(file_predictions, 'file')
        error('Predictions file not found: %s', file_predictions);
    end

    fprintf('  Loading: %s\n', file_analysis);
    load(file_analysis, 'analysisstruct');

    fprintf('  Loading: %s\n', file_predictions);
    load(file_predictions, 'animal_condition_identifier');

    % --- Build 66-feature matrix -----------------------------------------
    % jt_features:       N_good x 60   (joint features)
    % extra_jt_features: N_good x M    (extra features; take last 6 cols)
    jt   = analysisstruct.jt_features;
    extra = analysisstruct.extra_jt_features(:, end-5:end);
    D = [jt, extra];  % N_good x 66

    n_features = size(D, 2);
    fprintf('  Feature matrix: %d frames x %d features\n', size(D, 1), n_features);

    % Fill any NaN values with linear interpolation (same as Fig01)
    D = fillmissing(D, 'linear');

    % --- Map per-frame condition labels ----------------------------------
    % animal_condition_identifier is at the original frame rate.
    % Upsample by repfactor to match the wavelet-upsampled data,
    % then index by good-tracking frames.
    long_ids = repelem(animal_condition_identifier, repfactor);
    good_frames = analysisstruct.frames_with_good_tracking{1};
    frame_ids = long_ids(good_frames);  % same length as D rows

    % Extract condition letter (last character of each identifier)
    condition_letters = cellfun(@(x) x(end), frame_ids, 'UniformOutput', false);

    fprintf('  Total good-tracking frames: %d\n', length(frame_ids));

    % --- Split by condition and save -------------------------------------
    for k = 1:length(cohort.conditions)
        cond = cohort.conditions{k};
        cond_name = cohort.condition_names{k};

        mask = strcmp(condition_letters, cond);
        features = D(mask, :);

        fprintf('  Condition %s (%s): %d frames\n', cond, cond_name, size(features, 1));

        % Save .mat
        out_file = fullfile(output_dir, sprintf('%s_%s_features.mat', cohort.name, cond));
        save(out_file, 'features', '-v7.3');
        fprintf('  Saved: %s\n', out_file);
    end

    % Clean up large variables before next cohort
    clear analysisstruct animal_condition_identifier jt extra D long_ids frame_ids condition_letters;
end

fprintf('\n=== Feature extraction complete ===\n');
fprintf('Output directory: %s\n', output_dir);
