data_path = ('C:\Users\acuna\Repositories\LTSF-Linear\results\PCA_MA2_380_12_96_t_ma5_DLinear_custom_ftS_sl380_pl12_inTrue_it1_lr0.0001_bs32_movag1024_lablen96_freqt\');
x = dir('results\MA_96_12_Autoformer_custom_ftM_sl96_pl12_inTrue_it1_lr0.0001_bs32\');
preds =  py.numpy.load(fullfile(data_path,'pred.npy'));
trues =  py.numpy.load(fullfile(data_path,'true.npy'));
%
p = double(preds);
t = double(trues);
%
figure, plot(p(10000,:,1))
hold on
plot(t(10000,:,1))
legend({'p', 't'})
%%
data_path = ('C:\Users\acuna\Repositories\LTSF-Linear\results\PCA_MA_380_12_96_t_ma4_DLinear_custom_ftS_sl380_pl12_inTrue_it1_lr0.0001_bs32_movag1024_lablen96_freqt');
x = dir('results\MA_96_12_Autoformer_custom_ftM_sl96_pl12_inTrue_it1_lr0.0001_bs32\');
preds =  py.numpy.load(fullfile(data_path,'pred.npy'));
trues =  py.numpy.load(fullfile(data_path,'true.npy'));
%
p = double(preds);
t = double(trues);
%
figure, plot(p(10000,:,1))
hold on
plot(t(10000,:,1))
legend({'p', 't'})


%%
% Sequence-to-Sequence Model Visualization for Stride 1 Predictions vs. Ground Truth
% Assumes:
% - p: prediction array of shape (m, n, 1)
% - t: ground truth array of the same shape
% Where:
%   - m is the number of sequences (with stride 1)
%   - n is 12 (future frames per prediction)
%   - The last dimension is 1 (single-channel values)

% If you don't have p and t defined, uncomment this to generate sample data:
% m = 30;  % Number of sequences
% n = 12;  % Number of frames per sequence
% p = zeros(m, n, 1);
% t = zeros(m, n, 1);
% for i = 1:m
%     for j = 1:n
%         % Generate some pattern (simple sine wave with some noise for predictions)
%         t(i, j, 1) = 0.5 + 0.3 * sin((i+j)/5);  % Ground truth
%         p(i, j, 1) = t(i, j, 1) + 0.1 * randn();  % Predictions with noise
%     end
% end

% Get dimensions
[m, n, ~] = size(p);
clc
fprintf('Visualizing predictions vs. ground truth of shape (%d, %d, 1) with stride 1\n', m, n);

%% 1. Create a unified timeline visualization
% With stride 1, each sequence starts one frame after the previous
% This creates a unified timeline where we can see how predictions evolve

% Calculate how many total "real" frames we're representing
total_frames = m + n - 1;

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

%% 2. Calculate average prediction and ground truth for each position in the timeline
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

%% Create main figure: Average Prediction vs. Ground Truth
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Average prediction vs. ground truth
subplot(2, 1, 1);
hold on;
plot(1:total_frames, avg_prediction, 'b-', 'LineWidth', 2, 'DisplayName', 'Avg. Prediction');
plot(1:total_frames, avg_truth, 'r-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
title('Average Prediction vs. Ground Truth');
xlabel('Frame Position in Timeline');
ylabel('Value');
legend('Location', 'best');
grid on;
xlim([1, total_frames]);

% Add confidence interval for predictions (mean ± standard deviation)
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

% Plot confidence interval
x_region = 1:total_frames;
y_upper = avg_prediction + pred_std;
y_lower = avg_prediction - pred_std;
fill([x_region, fliplr(x_region)], [y_lower, fliplr(y_upper)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '±1 Std Dev');

% Plot 2: Deviation (prediction error)
subplot(2, 1, 2);
deviation = avg_prediction - avg_truth;
bar(1:total_frames, deviation);
title('Prediction Error (Prediction - Ground Truth)');
xlabel('Frame Position in Timeline');
ylabel('Error');
grid on;
xlim([0.5, total_frames+0.5]);

% Add a horizontal line at zero
hold on;
plot([0, total_frames+1], [0, 0], 'k--', 'LineWidth', 1);

% Color positive and negative deviations differently
for i = 1:total_frames
    if deviation(i) >= 0
        % Positive deviation (overestimation)
        bar(i, deviation(i), 'FaceColor', [0.8, 0.2, 0.2]);  % Red
    else
        % Negative deviation (underestimation)
        bar(i, deviation(i), 'FaceColor', [0.2, 0.2, 0.8]);  % Blue
    end
end

% Adjust layout
sgtitle('Comparison of Predictions vs. Ground Truth (Stride 1)', 'FontSize', 16);

%% Create second figure: Error metrics across timeline
figure('Position', [100, 100, 1200, 600]);

% Plot 1: Absolute Error
subplot(2, 2, 1);
abs_error = abs(avg_prediction - avg_truth);
bar(1:total_frames, abs_error);
title('Mean Absolute Error');
xlabel('Frame Position');
ylabel('|Error|');
grid on;
xlim([0.5, total_frames+0.5]);

% Plot 2: Squared Error
subplot(2, 2, 2);
sq_error = (avg_prediction - avg_truth).^2;
bar(1:total_frames, sq_error);
title('Mean Squared Error');
xlabel('Frame Position');
ylabel('Error^2');
grid on;
xlim([0.5, total_frames+0.5]);

% Plot 3: Relative Error (percentage)
subplot(2, 2, 3);
% Avoid division by zero
rel_error = abs_error ./ max(abs(avg_truth), 0.0001) * 100;
bar(1:total_frames, rel_error);
title('Relative Error (%)');
xlabel('Frame Position');
ylabel('Relative Error (%)');
grid on;
xlim([0.5, total_frames+0.5]);

% Plot 4: Number of predictions per position
subplot(2, 2, 4);
bar(1:total_frames, prediction_counts);
title('Number of Predictions per Position');
xlabel('Frame Position');
ylabel('Count');
grid on;
xlim([0.5, total_frames+0.5]);

% Adjust layout
sgtitle('Error Analysis Across Timeline', 'FontSize', 16);

%% Calculate and display overall error metrics
% Only calculate for positions with predictions
valid_positions = prediction_counts > 0;

% Mean Absolute Error (MAE)
mae = mean(abs_error(valid_positions));

% Root Mean Squared Error (RMSE)
rmse = sqrt(mean(sq_error(valid_positions)));

% Mean Absolute Percentage Error (MAPE)
mape = mean(rel_error(valid_positions));

fprintf('\nOverall Error Metrics:\n');
fprintf('Mean Absolute Error (MAE): %.4f\n', mae);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);
fprintf('Mean Absolute Percentage Error (MAPE): %.2f%%\n', mape);

% Optional: To save the figures
% saveas(gcf, 'prediction_vs_truth.png');

%% %%%%%%%%%%%%%%%%%%%%% %%
% Analysis of Sequence-to-Sequence Model Predictions by Prediction Horizon
% Assumes:
% - p: prediction array of shape (m, n, 1)
% - t: ground truth array of the same shape
% Where:
%   - m is the number of sequences (with stride 1)
%   - n is 12 (future frames per prediction)
%   - The last dimension is 1 (single-channel values)

% If you don't have p and t defined, uncomment this to generate sample data:
% m = 100;  % Number of sequences
% n = 12;   % Number of frames per sequence (prediction horizon)
% p = zeros(m, n, 1);
% t = zeros(m, n, 1);
% for i = 1:m
%     for j = 1:n
%         % True values follow a pattern
%         t(i, j, 1) = 0.5 + 0.3 * sin((i+j)/5);
%         
%         % Predictions get worse as horizon increases
%         noise_level = 0.05 * (j/2);  % More noise for further predictions
%         p(i, j, 1) = t(i, j, 1) + noise_level * randn();
%     end
% end

% Get dimensions
[m, n, ~] = size(p);
fprintf('Analyzing predictions by horizon for data of shape (%d, %d, 1)\n', m, n);

%% 1. Analyze error by prediction horizon (frame position)
% Initialize arrays to store error metrics for each horizon
mae_by_horizon = zeros(1, n);
rmse_by_horizon = zeros(1, n);
mape_by_horizon = zeros(1, n);
correlation_by_horizon = zeros(1, n);

% Calculate error metrics for each prediction horizon
for horizon = 1:n
    % Extract all predictions and ground truths for this horizon
    horizon_p = squeeze(p(:, horizon, 1));
    horizon_t = squeeze(t(:, horizon, 1));
    
    % Mean Absolute Error
    mae_by_horizon(horizon) = mean(abs(horizon_p - horizon_t));
    
    % Root Mean Squared Error
    rmse_by_horizon(horizon) = sqrt(mean((horizon_p - horizon_t).^2));
    
    % Mean Absolute Percentage Error (avoid division by zero)
    mape_by_horizon(horizon) = mean(abs(horizon_p - horizon_t) ./ max(abs(horizon_t), 0.0001)) * 100;
    
    % Correlation coefficient
    correlation_by_horizon(horizon) = corr(horizon_p, horizon_t);
end

%% Create main figure: Error metrics by prediction horizon
figure('Position', [100, 100, 1200, 800]);

% Plot 1: MAE by horizon
subplot(2, 2, 1);
bar(1:n, mae_by_horizon);
title('Mean Absolute Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('MAE');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 2: RMSE by horizon
subplot(2, 2, 2);
bar(1:n, rmse_by_horizon);
title('Root Mean Squared Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('RMSE');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 3: MAPE by horizon
subplot(2, 2, 3);
bar(1:n, mape_by_horizon);
title('Mean Absolute Percentage Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('MAPE (%)');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 4: Correlation by horizon
subplot(2, 2, 4);
bar(1:n, correlation_by_horizon);
title('Correlation between Predictions and Ground Truth');
xlabel('Prediction Horizon (Frame)');
ylabel('Correlation Coefficient');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);
ylim([0, 1]);  % Correlation usually between 0 and 1 for this type of data

% Adjust layout
sgtitle('Prediction Accuracy Analysis by Horizon', 'FontSize', 16);

%% Create second figure: Predictions vs. Ground Truth for different horizons
figure('Position', [100, 100, 1200, 800]);

% Select a few representative horizons to display
horizons_to_show = [1, 4, 8, 12];  % First, middle, and last frames
colors = jet(length(horizons_to_show));

% Plot predictions vs. ground truth for selected horizons
for idx = 1:length(horizons_to_show)
    h = horizons_to_show(idx);
    
    % Extract data for this horizon
    horizon_p = squeeze(p(:, h, 1));
    horizon_t = squeeze(t(:, h, 1));
    
    % Create scatter plot
    subplot(2, 2, idx);
    
    % First plot identity line (perfect prediction)
    min_val = min(min(horizon_p), min(horizon_t));
    max_val = max(max(horizon_p), max(horizon_t));
    range = [min_val - 0.1, max_val + 0.1];
    
    hold on;
    plot(range, range, 'k--', 'LineWidth', 1);
    
    % Then scatter plot of predictions vs. ground truth
    scatter(horizon_t, horizon_p, 30, colors(idx, :), 'filled', 'MarkerFaceAlpha', 0.6);
    
    title(sprintf('Horizon %d (Frame %d)', h, h));
    xlabel('Ground Truth');
    ylabel('Prediction');
    grid on;
    axis equal;
    axis([range, range]);
    
    % Add text with error metrics
    text(range(1) + 0.1*(range(2)-range(1)), range(2) - 0.1*(range(2)-range(1)), ...
        sprintf('MAE: %.4f\nRMSE: %.4f\nCorr: %.4f', ...
        mae_by_horizon(h), rmse_by_horizon(h), correlation_by_horizon(h)), ...
        'FontSize', 9);
end

% Adjust layout
sgtitle('Predictions vs. Ground Truth for Different Horizons', 'FontSize', 16);

%% Create third figure: Mean and distribution of predictions by horizon
figure('Position', [100, 100, 1200, 600]);

% Plot 1: Average predictions vs. ground truth by horizon
subplot(1, 2, 1);
hold on;

% Create x values for each horizon with slight offset for clarity
x_pred = (1:n) - 0.15;
x_truth = (1:n) + 0.15;

% Calculate mean values for each horizon
mean_p = zeros(1, n);
mean_t = zeros(1, n);
std_p = zeros(1, n);

for h = 1:n
    mean_p(h) = mean(squeeze(p(:, h, 1)));
    mean_t(h) = mean(squeeze(t(:, h, 1)));
    std_p(h) = std(squeeze(p(:, h, 1)));
end

% Bar chart with error bars
bar_p = bar(x_pred, mean_p, 0.3, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
bar_t = bar(x_truth, mean_t, 0.3, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% Add error bars for predictions
errorbar(x_pred, mean_p, std_p, '.k');

title('Average Prediction vs. Ground Truth by Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('Value');
legend([bar_p, bar_t], {'Prediction', 'Ground Truth'}, 'Location', 'best');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 2: Boxplot of prediction errors by horizon
subplot(1, 2, 2);
hold on;

% Prepare data for boxplot
errors_by_horizon = cell(1, n);
for h = 1:n
    errors_by_horizon{h} = squeeze(p(:, h, 1) - t(:, h, 1));
end

% Create boxplot
boxplot(cell2mat(errors_by_horizon), 'Colors', 'b', 'Symbol', 'r+');

title('Distribution of Prediction Errors by Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('Error (Prediction - Ground Truth)');
grid on;
ylim([-1, 1]);  % Adjust based on your data range

% Add a horizontal line at zero
yline(0, 'k--');

% Adjust layout
sgtitle('Statistical Analysis of Predictions by Horizon', 'FontSize', 16);

%% Print summary statistics
fprintf('\nSummary of Error Metrics by Prediction Horizon:\n');
fprintf('Horizon\tMAE\t\tRMSE\t\tMAPE\t\tCorrelation\n');
fprintf('-------\t-------\t\t-------\t\t-------\t\t-----------\n');
for h = 1:n
    fprintf('%d\t%.4f\t\t%.4f\t\t%.2f%%\t\t%.4f\n', ...
        h, mae_by_horizon(h), rmse_by_horizon(h), mape_by_horizon(h), correlation_by_horizon(h));
end

% Optional: To save the figures
% saveas(gcf, 'prediction_horizon_analysis.png');

%% %%%%%%%%%%%%%%%%%%%%%
% Analysis of Sequence-to-Sequence Model Predictions by Prediction Horizon
% Assumes:
% - p: prediction array of shape (m, n, 1)
% - t: ground truth array of the same shape
% Where:
%   - m is the number of sequences (with stride 1)
%   - n is 12 (future frames per prediction)
%   - The last dimension is 1 (single-channel values)

% If you don't have p and t defined, uncomment this to generate sample data:
% m = 100;  % Number of sequences
% n = 12;   % Number of frames per sequence (prediction horizon)
% p = zeros(m, n, 1);
% t = zeros(m, n, 1);
% for i = 1:m
%     for j = 1:n
%         % True values follow a pattern
%         t(i, j, 1) = 0.5 + 0.3 * sin((i+j)/5);
%         
%         % Predictions get worse as horizon increases
%         noise_level = 0.05 * (j/2);  % More noise for further predictions
%         p(i, j, 1) = t(i, j, 1) + noise_level * randn();
%     end
% end

% Get dimensions
[m, n, ~] = size(p);
fprintf('Analyzing predictions by horizon for data of shape (%d, %d, 1)\n', m, n);

%% 1. Analyze error by prediction horizon (frame position)
% Initialize arrays to store error metrics for each horizon
mae_by_horizon = zeros(1, n);
rmse_by_horizon = zeros(1, n);
mape_by_horizon = zeros(1, n);
correlation_by_horizon = zeros(1, n);

% Initialize arrays for Standard Error of the Mean (SEM)
mae_sem = zeros(1, n);
rmse_sem = zeros(1, n);
mape_sem = zeros(1, n);

% Calculate error metrics for each prediction horizon
for horizon = 1:n
    % Extract all predictions and ground truths for this horizon
    horizon_p = squeeze(p(:, horizon, 1));
    horizon_t = squeeze(t(:, horizon, 1));
    
    % Calculate individual errors for SEM
    abs_errors = abs(horizon_p - horizon_t);
    squared_errors = (horizon_p - horizon_t).^2;
    percentage_errors = abs_errors ./ max(abs(horizon_t), 0.0001) * 100;
    
    % Mean Absolute Error
    mae_by_horizon(horizon) = mean(abs_errors);
    
    % Root Mean Squared Error
    rmse_by_horizon(horizon) = sqrt(mean(squared_errors));
    
    % Mean Absolute Percentage Error (avoid division by zero)
    mape_by_horizon(horizon) = mean(percentage_errors);
    
    % Correlation coefficient
    correlation_by_horizon(horizon) = corr(horizon_p, horizon_t);
    
    % Standard Error of the Mean (SEM) for each metric
    % SEM = standard deviation / sqrt(sample size)
    mae_sem(horizon) = std(abs_errors) / sqrt(length(abs_errors));
    rmse_sem(horizon) = std(sqrt(squared_errors)) / sqrt(length(squared_errors));
    mape_sem(horizon) = std(percentage_errors) / sqrt(length(percentage_errors));
end

%% Create main figure: Error metrics by prediction horizon
figure('Position', [100, 100, 1200, 800]);

% Plot 1: MAE by horizon with SEM error bars
subplot(2, 2, 1);
b1 = bar(1:n, mae_by_horizon);
hold on;
% Add error bars for SEM
errorbar(1:n, mae_by_horizon, mae_sem, 'k.', 'LineWidth', 1.5);
title('Mean Absolute Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('MAE ± SEM');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 2: RMSE by horizon with SEM error bars
subplot(2, 2, 2);
b2 = bar(1:n, rmse_by_horizon);
hold on;
% Add error bars for SEM
errorbar(1:n, rmse_by_horizon, rmse_sem, 'k.', 'LineWidth', 1.5);
title('Root Mean Squared Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('RMSE ± SEM');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 3: MAPE by horizon with SEM error bars
subplot(2, 2, 3);
b3 = bar(1:n, mape_by_horizon);
hold on;
% Add error bars for SEM
errorbar(1:n, mape_by_horizon, mape_sem, 'k.', 'LineWidth', 1.5);
title('Mean Absolute Percentage Error by Prediction Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('MAPE (%) ± SEM');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 4: Correlation by horizon
subplot(2, 2, 4);
bar(1:n, correlation_by_horizon);
title('Correlation between Predictions and Ground Truth');
xlabel('Prediction Horizon (Frame)');
ylabel('Correlation Coefficient');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);
ylim([0, 1]);  % Correlation usually between 0 and 1 for this type of data

% Adjust layout
sgtitle('Prediction Accuracy Analysis by Horizon', 'FontSize', 16);

%% Create second figure: Predictions vs. Ground Truth for different horizons
figure('Position', [100, 100, 1200, 800]);

% Select a few representative horizons to display
horizons_to_show = [1, 4, 8, 12];  % First, middle, and last frames
colors = jet(length(horizons_to_show));

% Plot predictions vs. ground truth for selected horizons
for idx = 1:length(horizons_to_show)
    h = horizons_to_show(idx);
    
    % Extract data for this horizon
    horizon_p = squeeze(p(:, h, 1));
    horizon_t = squeeze(t(:, h, 1));
    
    % Create scatter plot
    subplot(2, 2, idx);
    
    % First plot identity line (perfect prediction)
    min_val = min(min(horizon_p), min(horizon_t));
    max_val = max(max(horizon_p), max(horizon_t));
    range = [min_val - 0.1, max_val + 0.1];
    
    hold on;
    plot(range, range, 'k--', 'LineWidth', 1);
    
    % Then scatter plot of predictions vs. ground truth
    scatter(horizon_t, horizon_p, 30, colors(idx, :), 'filled', 'MarkerFaceAlpha', 0.6);
    
    title(sprintf('Horizon %d (Frame %d)', h, h));
    xlabel('Ground Truth');
    ylabel('Prediction');
    grid on;
    axis equal;
    axis([range, range]);
    
    % Add text with error metrics including SEM
    text(range(1) + 0.1*(range(2)-range(1)), range(2) - 0.1*(range(2)-range(1)), ...
        sprintf('MAE: %.4f ± %.4f\nRMSE: %.4f ± %.4f\nMAPE: %.2f%% ± %.2f%%\nCorr: %.4f', ...
        mae_by_horizon(h), mae_sem(h), rmse_by_horizon(h), rmse_sem(h),... 
        mape_by_horizon(h), mape_sem(h), correlation_by_horizon(h)), ...
        'FontSize', 8);
end

% Adjust layout
sgtitle('Predictions vs. Ground Truth for Different Horizons', 'FontSize', 16);

%% Create third figure: Mean and distribution of predictions by horizon
figure('Position', [100, 100, 1200, 600]);

% Plot 1: Average predictions vs. ground truth by horizon
subplot(1, 2, 1);
hold on;

% Create x values for each horizon with slight offset for clarity
x_pred = (1:n) - 0.15;
x_truth = (1:n) + 0.15;

% Calculate mean values for each horizon
mean_p = zeros(1, n);
mean_t = zeros(1, n);
std_p = zeros(1, n);

for h = 1:n
    mean_p(h) = mean(squeeze(p(:, h, 1)));
    mean_t(h) = mean(squeeze(t(:, h, 1)));
    std_p(h) = std(squeeze(p(:, h, 1)));
end

% Calculate SEM for mean predictions
sem_p = std_p ./ sqrt(m);

% Bar chart with error bars
bar_p = bar(x_pred, mean_p, 0.3, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
bar_t = bar(x_truth, mean_t, 0.3, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% Add error bars using SEM instead of std
errorbar(x_pred, mean_p, sem_p, '.k', 'LineWidth', 1.5);

title('Average Prediction vs. Ground Truth by Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('Value');
legend([bar_p, bar_t], {'Prediction', 'Ground Truth'}, 'Location', 'best');
grid on;
xlim([0.5, n+0.5]);
xticks(1:n);

% Plot 2: Boxplot of prediction errors by horizon
subplot(1, 2, 2);
hold on;

% Prepare data for boxplot
errors_by_horizon = cell(1, n);
for h = 1:n
    errors_by_horizon{h} = squeeze(p(:, h, 1) - t(:, h, 1));
end

% Create boxplot
boxplot(cell2mat(errors_by_horizon), 'Colors', 'b', 'Symbol', 'r+');

title('Distribution of Prediction Errors by Horizon');
xlabel('Prediction Horizon (Frame)');
ylabel('Error (Prediction - Ground Truth)');
grid on;
ylim([-1, 1]);  % Adjust based on your data range

% Add a horizontal line at zero
yline(0, 'k--');

% Adjust layout
sgtitle('Statistical Analysis of Predictions by Horizon', 'FontSize', 16);

%% Print summary statistics
fprintf('\nSummary of Error Metrics by Prediction Horizon (with SEM):\n');
fprintf('Horizon\tMAE ± SEM\t\tRMSE ± SEM\t\tMAPE ± SEM\t\tCorrelation\n');
fprintf('-------\t--------------\t\t--------------\t\t--------------\t\t-----------\n');
for h = 1:n
    fprintf('%d\t%.4f ± %.4f\t\t%.4f ± %.4f\t\t%.2f%% ± %.2f%%\t\t%.4f\n', ...
        h, mae_by_horizon(h), mae_sem(h), rmse_by_horizon(h), rmse_sem(h),... 
        mape_by_horizon(h), mape_sem(h), correlation_by_horizon(h));
end

% Optional: To save the figures
% saveas(gcf, 'prediction_horizon_analysis.png');