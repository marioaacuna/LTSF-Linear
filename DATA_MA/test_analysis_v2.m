% Define the experimental conditions to analyze
conditions = struct();
rootpath = fullfile(pwd, 'results');
conditions(1).name = 'Baseline';
conditions(1).data_path = fullfile(rootpath,'TRAINING_BASELINE');
conditions(2).name = 'Formalin';
conditions(2).data_path = fullfile(rootpath,'formalin');

% Initialize arrays to store metrics for each condition
num_conditions = length(conditions);
all_preds = cell(1, num_conditions);
all_trues = cell(1, num_conditions);
all_p = cell(1, num_conditions);
all_t = cell(1, num_conditions);

% Set color scheme for visualizations
condition_colors = lines(num_conditions);

% Loop through each condition and load data
for c = 1:num_conditions
    fprintf('Loading data for condition: %s\n', conditions(c).name);
    data_path = conditions(c).data_path;
    
    % Load data
    all_preds{c} = py.numpy.load(fullfile(data_path, 'pred.npy'));
    all_trues{c} = py.numpy.load(fullfile(data_path, 'true.npy'));
    
    % Convert to MATLAB format
    all_p{c} = double(all_preds{c});
    all_t{c} = double(all_trues{c});
    
    % Get dimensions to verify
    [m, n, ~] = size(all_p{c});
    fprintf('Data shape for %s: (%d, %d, 1)\n', conditions(c).name, m, n);
end

%%
% Analyze prediction metrics across multiple experimental conditions
% For each condition, calculates and compares:
%   - MAE (Mean Absolute Error)
%   - RMSE (Root Mean Squared Error) 
%   - MAPE (Mean Absolute Percentage Error)
%   - Correlation between predictions and ground truth

% Initialize matrices to store metrics by horizon for each condition
num_horizons = size(all_p{1}, 2); % Assuming all conditions have same number of horizons
mae_by_condition = zeros(num_conditions, num_horizons);
rmse_by_condition = zeros(num_conditions, num_horizons);
mape_by_condition = zeros(num_conditions, num_horizons);
corr_by_condition = zeros(num_conditions, num_horizons);

% Standard Error of Mean (SEM)
mae_sem_by_condition = zeros(num_conditions, num_horizons);
rmse_sem_by_condition = zeros(num_conditions, num_horizons);
mape_sem_by_condition = zeros(num_conditions, num_horizons);

% Calculate metrics for each condition and horizon
for c = 1:num_conditions
    p = all_p{c};
    t = all_t{c};
    [m, n, ~] = size(p);
    
    for horizon = 1:n
        % Extract predictions and ground truths for this horizon
        horizon_p = squeeze(p(:, horizon, 1));
        horizon_t = squeeze(t(:, horizon, 1));
        
        % Calculate individual errors for SEM
        abs_errors = abs(horizon_p - horizon_t);
        squared_errors = (horizon_p - horizon_t).^2;
        percentage_errors = abs_errors ./ max(abs(horizon_t), 0.0001) * 100;
        
        % Mean Absolute Error
        mae_by_condition(c, horizon) = mean(abs_errors);
        
        % Root Mean Squared Error
        rmse_by_condition(c, horizon) = sqrt(mean(squared_errors));
        
        % Mean Absolute Percentage Error (avoid division by zero)
        mape_by_condition(c, horizon) = mean(percentage_errors);
        
        % Correlation coefficient
        corr_by_condition(c, horizon) = corr(horizon_p, horizon_t);
        
        % Standard Error of the Mean (SEM) for each metric
        mae_sem_by_condition(c, horizon) = std(abs_errors) / sqrt(length(abs_errors));
        rmse_sem_by_condition(c, horizon) = std(sqrt(squared_errors)) / sqrt(length(squared_errors));
        mape_sem_by_condition(c, horizon) = std(percentage_errors) / sqrt(length(percentage_errors));
    end
end

%% Compare error metrics directly between conditions with line plots
figure('Position', [100, 100, 1200, 800]);

% MAE comparison
subplot(2, 2, 1);
hold on;
for c = 1:num_conditions
    errorbar(1:num_horizons, mae_by_condition(c, :), mae_sem_by_condition(c, :), ...
        'Color', condition_colors(c,:), 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', condition_colors(c,:), 'DisplayName', conditions(c).name);
end
title('Mean Absolute Error Comparison');
xlabel('Prediction Horizon (Frame)');
ylabel('MAE ± SEM');
legend('Location', 'northwest');
grid on;
xlim([0.5, num_horizons+0.5]);
xticks(1:num_horizons);

% RMSE comparison
subplot(2, 2, 2);
hold on;
for c = 1:num_conditions
    errorbar(1:num_horizons, rmse_by_condition(c, :), rmse_sem_by_condition(c, :), ...
        'Color', condition_colors(c,:), 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', condition_colors(c,:), 'DisplayName', conditions(c).name);
end
title('Root Mean Squared Error Comparison');
xlabel('Prediction Horizon (Frame)');
ylabel('RMSE ± SEM');
legend('Location', 'northwest');
grid on;
xlim([0.5, num_horizons+0.5]);
xticks(1:num_horizons);

% MAPE comparison
subplot(2, 2, 3);
hold on;
for c = 1:num_conditions
    errorbar(1:num_horizons, mape_by_condition(c, :), mape_sem_by_condition(c, :), ...
        'Color', condition_colors(c,:), 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', condition_colors(c,:), 'DisplayName', conditions(c).name);
end
title('Mean Absolute Percentage Error Comparison');
xlabel('Prediction Horizon (Frame)');
ylabel('MAPE (%) ± SEM');
legend('Location', 'northwest');
grid on;
xlim([0.5, num_horizons+0.5]);
xticks(1:num_horizons);

% Correlation comparison
subplot(2, 2, 4);
hold on;
for c = 1:num_conditions
    plot(1:num_horizons, corr_by_condition(c, :), ...
        'Color', condition_colors(c,:), 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', condition_colors(c,:), 'DisplayName', conditions(c).name);
end
title('Correlation Coefficient Comparison');
xlabel('Prediction Horizon (Frame)');
ylabel('Correlation');
legend('Location', 'southwest');
grid on;
xlim([0.5, num_horizons+0.5]);
xticks(1:num_horizons);
ylim([0, 1]);

% Adjust layout
sgtitle('Direct Comparison of Error Metrics Between Conditions', 'FontSize', 16);

%% Print summary statistics for all conditions
fprintf('\n======= SUMMARY OF PREDICTION METRICS BY CONDITION =======\n\n');

% Calculate global metrics for each condition
global_mae = mean(mae_by_condition, 2);
global_rmse = mean(rmse_by_condition, 2);
global_mape = mean(mape_by_condition, 2);
global_corr = mean(corr_by_condition, 2);

% Print global metrics table
fprintf('GLOBAL METRICS (averaged across all horizons):\n');
fprintf('Condition\tMAE\t\tRMSE\t\tMAPE\t\tCorrelation\n');
fprintf('----------\t-------\t\t-------\t\t-------\t\t-----------\n');
for c = 1:num_conditions
    fprintf('%s\t%.4f\t\t%.4f\t\t%.2f%%\t\t%.4f\n', ...
        conditions(c).name, global_mae(c), global_rmse(c), global_mape(c), global_corr(c));
end
fprintf('\n');

% Print detailed metrics by horizon for each condition
for c = 1:num_conditions
    fprintf('\nDETAILED METRICS FOR %s:\n', conditions(c).name);
    fprintf('Horizon\tMAE ± SEM\t\tRMSE ± SEM\t\tMAPE ± SEM\t\tCorrelation\n');
    fprintf('-------\t--------------\t\t--------------\t\t--------------\t\t-----------\n');
    for h = 1:num_horizons
        fprintf('%d\t%.4f ± %.4f\t\t%.4f ± %.4f\t\t%.2f%% ± %.2f%%\t\t%.4f\n', ...
            h, mae_by_condition(c, h), mae_sem_by_condition(c, h), ...
            rmse_by_condition(c, h), rmse_sem_by_condition(c, h), ...
            mape_by_condition(c, h), mape_sem_by_condition(c, h), ...
            corr_by_condition(c, h));
    end
    fprintf('\n');
end

%% Create unified timeline visualizations for each condition
% Store the average predictions and metrics for comparison
all_avg_predictions = cell(num_conditions, 1);
all_avg_truths = cell(num_conditions, 1);
all_deviations = cell(num_conditions, 1);
all_pred_stds = cell(num_conditions, 1);
all_total_frames = zeros(num_conditions, 1);

for c = 1:num_conditions
    % Set the current data to the selected condition
    p = all_p{c};
    t = all_t{c};
    
    % Get dimensions
    [m, n, ~] = size(p);
    
    % Calculate how many total "real" frames we're representing
    total_frames = m + n - 1;
    all_total_frames(c) = total_frames;
    
    % Create matrices to store all predictions and ground truth mapped to their timeline positions
    unified_p = NaN(m, total_frames);
    unified_t = NaN(m, total_frames);
    
    % Fill the matrices with predictions and ground truth
    for seq = 1:m
        for frame = 1:n
            % Map each value to its position in the unified timeline
            timeline_pos = seq + frame - 1;
            unified_p(seq, timeline_pos) = p(seq, frame, 1);
            unified_t(seq, timeline_pos) = t(seq, frame, 1);
        end
    end
    
    % Calculate average prediction and ground truth for each position in the timeline
    avg_prediction = zeros(1, total_frames);
    avg_truth = zeros(1, total_frames);
    prediction_counts = zeros(1, total_frames);
    
    % For each real position in the timeline, gather all predictions and ground truth
    for seq = 1:m
        for frame = 1:n
            timeline_pos = seq + frame - 1;
            % Avoid NaN values
            if ~isnan(p(seq, frame, 1))
                avg_prediction(timeline_pos) = avg_prediction(timeline_pos) + p(seq, frame, 1);
                avg_truth(timeline_pos) = avg_truth(timeline_pos) + t(seq, frame, 1);
                prediction_counts(timeline_pos) = prediction_counts(timeline_pos) + 1;
            end
        end
    end
    
    % Calculate averages
    avg_prediction = avg_prediction ./ max(prediction_counts, 1);  % Avoid division by zero
    avg_truth = avg_truth ./ max(prediction_counts, 1);
    
    % Calculate confidence interval for predictions (mean ± standard deviation)
    pred_std = zeros(1, total_frames);
    for timeline_pos = 1:total_frames
        % Gather all predictions for this position
        all_preds = [];
        for seq = 1:m
            for frame = 1:n
                if seq + frame - 1 == timeline_pos && ~isnan(p(seq, frame, 1))
                    all_preds = [all_preds, p(seq, frame, 1)];
                end
            end
        end
        
        % Calculate std if we have enough predictions
        if length(all_preds) > 1
            pred_std(timeline_pos) = std(all_preds);
        end
    end
    
    % Store results for comparison
    all_avg_predictions{c} = avg_prediction;
    all_avg_truths{c} = avg_truth;
    all_deviations{c} = avg_prediction - avg_truth;
    all_pred_stds{c} = pred_std;
end

%% Create zoomed-in comparison with vertical subplots for each condition
figure('Position', [100, 100, 1200, num_conditions*300]);

% Set the zoom range
zoom_range = min(3000, max(all_total_frames));
zoom_start = 15000;
zoom_end = zoom_start + zoom_range - 1;

% Create a subplot for each condition
for c = 1:num_conditions
    subplot(num_conditions, 1, c);
    hold on;
    
    total_frames = all_total_frames(c);
    avg_prediction = all_avg_predictions{c};
    avg_truth = all_avg_truths{c};
    pred_std = all_pred_stds{c};
    
    % Handle the case where this condition has fewer than zoom_end frames
    actual_end = min(zoom_end, total_frames);
    
    % Plot prediction (zoomed)
    plot(zoom_start:actual_end, avg_prediction(zoom_start:actual_end), 'LineWidth', 2, ...
        'Color', condition_colors(c,:), 'LineStyle', '-', ...
        'DisplayName', 'Prediction');
    
    % Plot ground truth (zoomed)
    plot(zoom_start:actual_end, avg_truth(zoom_start:actual_end), 'LineWidth', 2, ...
        'Color', 'black', 'LineStyle', '--', 'LineWidth', 0.5, ...
        'DisplayName', 'Ground Truth');
    
    % Plot confidence interval (zoomed)
    if zoom_start <= actual_end
        x_region = zoom_start:actual_end;
        y_upper = avg_prediction(zoom_start:actual_end) + pred_std(zoom_start:actual_end);
        y_lower = avg_prediction(zoom_start:actual_end) - pred_std(zoom_start:actual_end);
        fill([x_region, fliplr(x_region)], [y_lower, fliplr(y_upper)], condition_colors(c,:), ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
            'DisplayName', 'Std Dev');
    end
    
    title([conditions(c).name, ': Prediction vs. Ground Truth']);
    xlabel('Frame Position in Timeline');
    ylabel('Value');
    legend('Location', 'best');
    grid on;
    
    % Set fixed y-axis limits between -2 and 2 for all subplots
    ylim([-2, 2]);
    
    % Set x-axis limits based on zoom range
    xlim([zoom_start, zoom_end]);
    
    % Add text with metrics (adjust position based on fixed y-axis)
    text(zoom_start + 0.05 * zoom_range, 1.5, ...
        sprintf('MAE: %.4f\nRMSE: %.4f\nMAPE: %.2f%%\nCorr: %.4f', ...
        global_mae(c), global_rmse(c), global_mape(c), global_corr(c)), ...
        'FontSize', 9, 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'Margin', 3);
end

% Adjust layout
sgtitle('Zoomed Comparison of Predictions vs. Ground Truth (First 3000 Frames)', 'FontSize', 16);