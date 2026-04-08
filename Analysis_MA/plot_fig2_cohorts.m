%% plot_fig2_cohorts.m
% Figure 2 draft — DLinear prediction analysis separated by cohort.
%
% Generates per-cohort comparison plots (Control vs Pain):
%   Cohort 1 — SFC: S (Sham) vs F (Formalin)
%   Cohort 2 — HNG: H (Hargreaves) vs N (Neuropathic)
%
% For each cohort:
%   1. Error metrics by prediction horizon (MAE, RMSE, MAPE, Correlation)
%   2. Scatter: Predictions vs Ground Truth at selected horizons
%   3. Zoomed timeline: avg prediction vs ground truth with confidence bands
%
% Loads pred.npy / true.npy saved by run_all_conditions.py.
%
% Usage:
%   cd <repo_root>
%   run('Analysis_MA/plot_fig2_cohorts.m')

%% INIT ===================================================================
clc; clear; close all;

% Repo root (one level up from this script)
repo_root = fileparts(fileparts(mfilename('fullpath')));

% Results root
results_root = fullfile(repo_root, 'results');

% Output folder for exported figures
fig_output = fullfile(repo_root, 'Analysis_MA', 'figures');
if ~exist(fig_output, 'dir'), mkdir(fig_output); end

%% Publication-quality defaults
set(groot, 'defaultFigureColor', 'white');
set(groot, 'defaultAxesColor', 'white');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultTextColor', 'black');
set(groot, 'defaultAxesXColor', 'black');
set(groot, 'defaultAxesYColor', 'black');

%% Define cohorts and conditions ==========================================
% Model parameters (must match run_all_conditions.py)
seq_len  = 96;
pred_len = 12;

cohorts = struct();

cohorts(1).name         = 'SFC';
cohorts(1).display_name = 'SFC — Sham vs Formalin';
cohorts(1).conditions   = {'S', 'F'};
cohorts(1).cond_labels  = {'Sham (Control)', 'Formalin (Pain)'};
cohorts(1).cond_colors  = [0.2 0.6 0.9; 0.9 0.2 0.2];

cohorts(2).name         = 'HNG';
cohorts(2).display_name = 'HNG — Hargreaves vs Neuropathic';
cohorts(2).conditions   = {'H', 'N'};
cohorts(2).cond_labels  = {'Hargreaves (Control)', 'Neuropathic (Pain)'};
cohorts(2).cond_colors  = [0.2 0.7 0.4; 0.8 0.4 0.0];

%% Helper: build setting string (must match run_all_conditions.py) ========
build_setting = @(tag) sprintf( ...
    'MA_%s_%d_%d_DLinear_custom_ftM_sl%d_pl%d_inTrue_it1_lr0.0001_bs32', ...
    tag, seq_len, pred_len, seq_len, pred_len);

%% ========================================================================
%  MAIN LOOP — one full set of figures per cohort
%% ========================================================================
for ch = 1:length(cohorts)
    cohort = cohorts(ch);
    num_conds = length(cohort.conditions);

    fprintf('\n=== Cohort: %s ===\n', cohort.display_name);

    % --- Load predictions ------------------------------------------------
    all_p = cell(1, num_conds);
    all_t = cell(1, num_conds);

    for c = 1:num_conds
        tag = sprintf('%s_%s', cohort.name, cohort.conditions{c});
        setting = build_setting(tag);
        pred_file = fullfile(results_root, setting, 'pred.npy');
        true_file = fullfile(results_root, setting, 'true.npy');

        if ~exist(pred_file, 'file')
            error('Predictions not found: %s\nRun run_all_conditions.py first.', pred_file);
        end

        fprintf('  Loading %s ... ', tag);
        all_p{c} = double(py.numpy.load(pred_file));
        all_t{c} = double(py.numpy.load(true_file));
        [m, n, nf] = size(all_p{c});
        fprintf('shape (%d, %d, %d)\n', m, n, nf);
    end

    num_horizons = size(all_p{1}, 2);
    num_features = size(all_p{1}, 3);

    % =====================================================================
    %  Compute per-horizon metrics (averaged across ALL features)
    % =====================================================================
    mae_cond  = zeros(num_conds, num_horizons);
    rmse_cond = zeros(num_conds, num_horizons);
    mape_cond = zeros(num_conds, num_horizons);
    corr_cond = zeros(num_conds, num_horizons);

    mae_sem   = zeros(num_conds, num_horizons);
    rmse_sem  = zeros(num_conds, num_horizons);
    mape_sem  = zeros(num_conds, num_horizons);

    for c = 1:num_conds
        p = all_p{c};
        t = all_t{c};

        for h = 1:num_horizons
            % Flatten across all features for this horizon
            ph = reshape(p(:, h, :), [], 1);
            th = reshape(t(:, h, :), [], 1);

            abs_err  = abs(ph - th);
            sq_err   = (ph - th).^2;
            pct_err  = abs_err ./ max(abs(th), 1e-4) * 100;

            mae_cond(c, h)  = mean(abs_err);
            rmse_cond(c, h) = sqrt(mean(sq_err));
            mape_cond(c, h) = mean(pct_err);
            corr_cond(c, h) = corr(ph, th);

            mae_sem(c, h)  = std(abs_err)        / sqrt(numel(abs_err));
            rmse_sem(c, h) = std(sqrt(sq_err))    / sqrt(numel(sq_err));
            mape_sem(c, h) = std(pct_err)         / sqrt(numel(pct_err));
        end
    end

    % Global (avg across horizons) for annotations
    global_mae  = mean(mae_cond, 2);
    global_rmse = mean(rmse_cond, 2);
    global_mape = mean(mape_cond, 2);
    global_corr = mean(corr_cond, 2);

    % =====================================================================
    %  FIGURE 1: Error metrics by prediction horizon (line plots w/ SEM)
    % =====================================================================
    fig1 = figure('Position', [50, 100, 1200, 800]);

    metric_data  = {mae_cond, rmse_cond, mape_cond, corr_cond};
    metric_sem   = {mae_sem,  rmse_sem,  mape_sem,  []};
    metric_title = {'Mean Absolute Error', 'Root Mean Squared Error', ...
                    'Mean Abs. Percentage Error (%)', 'Correlation'};
    metric_ylab  = {'MAE \pm SEM', 'RMSE \pm SEM', 'MAPE (%) \pm SEM', 'Correlation'};

    for mi = 1:4
        subplot(2, 2, mi); hold on;
        for c = 1:num_conds
            if ~isempty(metric_sem{mi})
                errorbar(1:num_horizons, metric_data{mi}(c,:), metric_sem{mi}(c,:), ...
                    'Color', cohort.cond_colors(c,:), 'LineWidth', 2, ...
                    'Marker', 'o', 'MarkerSize', 6, ...
                    'MarkerFaceColor', cohort.cond_colors(c,:), ...
                    'DisplayName', cohort.cond_labels{c});
            else
                plot(1:num_horizons, metric_data{mi}(c,:), ...
                    'Color', cohort.cond_colors(c,:), 'LineWidth', 2, ...
                    'Marker', 'o', 'MarkerSize', 6, ...
                    'MarkerFaceColor', cohort.cond_colors(c,:), ...
                    'DisplayName', cohort.cond_labels{c});
            end
        end
        title(metric_title{mi});
        xlabel('Prediction Horizon (frame)');
        ylabel(metric_ylab{mi});
        legend('Location', 'best', 'Box', 'off');
        grid on; box off;
        xlim([0.5, num_horizons + 0.5]); xticks(1:num_horizons);
        if mi == 4, ylim([0 1]); end
    end

    sgtitle(sprintf('%s — Error Metrics by Horizon', cohort.display_name), ...
        'FontSize', 14, 'FontWeight', 'bold');

    exportgraphics(fig1, fullfile(fig_output, ...
        sprintf('Fig02_%s_error_metrics.pdf', cohort.name)), 'ContentType', 'vector');
    fprintf('  Exported: Fig02_%s_error_metrics.pdf\n', cohort.name);

    % =====================================================================
    %  FIGURE 2: Scatter — Preds vs Truth at selected horizons (feature 1)
    % =====================================================================
    horizons_to_show = unique(round(linspace(1, num_horizons, 4)));
    feat_idx = 1;  % use first feature for scatter

    fig2 = figure('Position', [50, 50, 1200, 800]);
    for hi = 1:length(horizons_to_show)
        hz = horizons_to_show(hi);
        subplot(2, ceil(length(horizons_to_show)/2), hi); hold on;

        for c = 1:num_conds
            ph = squeeze(all_p{c}(:, hz, feat_idx));
            th = squeeze(all_t{c}(:, hz, feat_idx));
            scatter(th, ph, 8, cohort.cond_colors(c,:), 'filled', ...
                'MarkerFaceAlpha', 0.3, 'DisplayName', cohort.cond_labels{c});
        end

        % Identity line
        ax_lim = [min(xlim(gca)), max(xlim(gca))];
        plot(ax_lim, ax_lim, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

        title(sprintf('Horizon %d', hz));
        xlabel('Ground Truth'); ylabel('Prediction');
        legend('Location', 'best', 'Box', 'off');
        grid on; box off; axis equal tight;
    end

    sgtitle(sprintf('%s — Prediction vs Truth (Feature %d)', ...
        cohort.display_name, feat_idx), 'FontSize', 14, 'FontWeight', 'bold');

    exportgraphics(fig2, fullfile(fig_output, ...
        sprintf('Fig02_%s_scatter.pdf', cohort.name)), 'ContentType', 'vector');
    fprintf('  Exported: Fig02_%s_scatter.pdf\n', cohort.name);

    % =====================================================================
    %  FIGURE 3: Zoomed timeline — avg prediction vs ground truth
    % =====================================================================
    % Use first feature for timeline visualisation
    fig3 = figure('Position', [100, 50, 1200, num_conds * 300]);

    for c = 1:num_conds
        p = all_p{c};
        t = all_t{c};
        [m, n, ~] = size(p);
        total_frames = m + n - 1;

        % Build unified average prediction & truth for feat_idx
        avg_pred  = zeros(1, total_frames);
        avg_truth = zeros(1, total_frames);
        counts    = zeros(1, total_frames);

        for seq = 1:m
            for fr = 1:n
                pos = seq + fr - 1;
                avg_pred(pos)  = avg_pred(pos)  + p(seq, fr, feat_idx);
                avg_truth(pos) = avg_truth(pos) + t(seq, fr, feat_idx);
                counts(pos)    = counts(pos) + 1;
            end
        end
        avg_pred  = avg_pred  ./ max(counts, 1);
        avg_truth = avg_truth ./ max(counts, 1);

        % Prediction std
        pred_std = zeros(1, total_frames);
        for pos = 1:total_frames
            vals = [];
            for seq = 1:m
                fr = pos - seq + 1;
                if fr >= 1 && fr <= n
                    vals(end+1) = p(seq, fr, feat_idx); %#ok<SAGROW>
                end
            end
            if numel(vals) > 1, pred_std(pos) = std(vals); end
        end

        % Zoom window
        zoom_len   = min(3000, total_frames);
        zoom_start = min(15000, total_frames - zoom_len + 1);
        zoom_start = max(zoom_start, 1);
        zoom_end   = zoom_start + zoom_len - 1;

        subplot(num_conds, 1, c); hold on;

        xr = zoom_start:zoom_end;
        plot(xr, avg_pred(xr),  'Color', cohort.cond_colors(c,:), 'LineWidth', 1.5, ...
            'DisplayName', 'Prediction');
        plot(xr, avg_truth(xr), 'k--', 'LineWidth', 0.5, ...
            'DisplayName', 'Ground Truth');

        % Confidence band
        y_up  = avg_pred(xr) + pred_std(xr);
        y_low = avg_pred(xr) - pred_std(xr);
        fill([xr, fliplr(xr)], [y_low, fliplr(y_up)], cohort.cond_colors(c,:), ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '\pm 1 SD');

        title(sprintf('%s', cohort.cond_labels{c}));
        xlabel('Frame'); ylabel('Value');
        legend('Location', 'best', 'Box', 'off');
        grid on; box off;
        ylim([-2 2]); xlim([zoom_start zoom_end]);

        % Annotation box
        text(zoom_start + 0.02*zoom_len, 1.5, ...
            sprintf('MAE: %.4f\nRMSE: %.4f\nCorr: %.4f', ...
            global_mae(c), global_rmse(c), global_corr(c)), ...
            'FontSize', 9, 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'Margin', 3);
    end

    sgtitle(sprintf('%s — Zoomed Prediction Timeline (Feature %d)', ...
        cohort.display_name, feat_idx), 'FontSize', 14, 'FontWeight', 'bold');

    exportgraphics(fig3, fullfile(fig_output, ...
        sprintf('Fig02_%s_timeline.pdf', cohort.name)), 'ContentType', 'vector');
    fprintf('  Exported: Fig02_%s_timeline.pdf\n', cohort.name);

    % =====================================================================
    %  Print summary table
    % =====================================================================
    fprintf('\n  %-25s  %8s  %8s  %8s  %8s\n', 'Condition', 'MAE', 'RMSE', 'MAPE%', 'Corr');
    fprintf('  %-25s  %8s  %8s  %8s  %8s\n', '---------', '---', '----', '-----', '----');
    for c = 1:num_conds
        fprintf('  %-25s  %8.4f  %8.4f  %7.2f%%  %8.4f\n', ...
            cohort.cond_labels{c}, global_mae(c), global_rmse(c), global_mape(c), global_corr(c));
    end
end

fprintf('\n=== All figures exported to: %s ===\n', fig_output);
