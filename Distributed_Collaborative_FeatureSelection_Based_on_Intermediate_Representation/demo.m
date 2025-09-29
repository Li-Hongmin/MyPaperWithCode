%% Distributed Collaborative Feature Selection - Unified Demo
% This demo supports both Leukemia and MNIST datasets
% Compares CFS (Collaborative) vs LFS (Local) feature selection methods
% Reproduces figures similar to experiment .fig files

clear; clc; close all;

% Setup paths for organized project structure
setup_paths();

%% 1. Dataset Selection
fprintf('=== Distributed Collaborative Feature Selection (DCFS) Demo ===\n');
fprintf('This demo compares Collaborative Feature Selection (CFS) vs Local Feature Selection (LFS)\n');
fprintf('Available datasets:\n');
fprintf('  1. Leukemia (gene expression data, 7129 features, 72 samples)\n');
fprintf('  2. MNIST (handwritten digits, 784 features, 70000 samples)\n\n');

% Dataset selection - check if running in batch mode
if exist('dataset_choice', 'var') && ~isempty(dataset_choice)
    % Use predefined choice (for batch mode or testing)
    fprintf('Using predefined dataset choice: %d\n', dataset_choice);
else
    % Interactive mode
    try
        dataset_choice = input('Select dataset (1=Leukemia, 2=MNIST): ');
    catch
        % Default to Leukemia for batch mode
        dataset_choice = 1;
        fprintf('Running in batch mode - defaulting to Leukemia dataset\n');
    end
end

if dataset_choice == 1
    dataset_name = 'Leukemia';
    data_path = './DATA_SET/leukemia/';
    exp_path = './experiments/leukemia/';
elseif dataset_choice == 2
    dataset_name = 'MNIST';
    data_path = './DATA_SET/MNIST/';
    exp_path = './experiments/mnist/';
else
    error('Invalid dataset choice. Please select 1 or 2.');
end

fprintf('\n✓ Selected dataset: %s\n\n', dataset_name);

%% 2. Load Dataset
fprintf('Step 1: Loading %s dataset...\n', dataset_name);

if dataset_choice == 1
    % Load Leukemia dataset
    try
        load([data_path 'leukemia_data.mat']);
        load([data_path 'leukemia_label.mat']);

        X_train = data.train';    % Features × samples format
        X_test = data.test';
        y_train = label.train';
        y_test = label.test';

        fprintf('✓ Leukemia data loaded successfully\n');
    catch
        error('❌ Leukemia data files not found');
    end

elseif dataset_choice == 2
    % Load MNIST dataset (subset for demo)
    try
        load([data_path 'Orig.mat']);

        % Use subset for demo (first 1000 samples, first 200 features)
        n_samples = min(1000, size(fea, 1));
        n_features = min(200, size(fea, 2));

        % Split into train/test (70/30 split)
        train_size = round(0.7 * n_samples);

        X_all = fea(1:n_samples, 1:n_features)';  % Features × samples format
        y_all = gnd(1:n_samples)';

        X_train = X_all(:, 1:train_size);
        X_test = X_all(:, (train_size+1):end);
        y_train = y_all(1:train_size);
        y_test = y_all((train_size+1):end);

        fprintf('✓ MNIST subset loaded successfully\n');
    catch
        error('❌ MNIST data files not found');
    end
end

fprintf('  - Training set: %d features × %d samples\n', size(X_train,1), size(X_train,2));
fprintf('  - Test set: %d features × %d samples\n', size(X_test,1), size(X_test,2));
fprintf('  - Number of classes: %d\n\n', length(unique(y_train)));

%% 3. Configure Algorithm Parameters
fprintf('Step 2: Setting up algorithm parameters...\n');

% Distributed settings
num_divisions = 2;
num_anchors = min(30, size(X_train, 2) - 5);  % Ensure anchors < samples

% Algorithm parameters (optimized per dataset)
if dataset_choice == 1
    % Leukemia parameters
    param.k = 3;
    param.delta1 = 5e-3;
    param.delta2 = 2.0;
    param.delta3 = 1.35;
    param.neig = 12;
else
    % MNIST parameters
    param.k = 10;
    param.delta1 = 1e-3;
    param.delta2 = 1.5;
    param.delta3 = 1.2;
    param.neig = 15;
end

param.na = num_anchors;
param.kernel = 'L';
param.ksigma = 0;

fprintf('  - Dataset: %s\n', dataset_name);
fprintf('  - Distributed nodes: %d\n', num_divisions);
fprintf('  - Number of anchors: %d\n', num_anchors);
fprintf('  - Kernel type: %s (Linear)\n\n', param.kernel);

%% 4. Create Results Storage
fprintf('Step 3: Preparing results storage...\n');

% Create figures directory if it doesn't exist
if ~exist('./figures', 'dir')
    mkdir('./figures');
end

% Initialize result storage
results = struct();
results.dataset = dataset_name;
results.parameters = param;
results.num_divisions = num_divisions;

fprintf('✓ Results storage prepared\n\n');

%% 5. Run Collaborative Feature Selection (CFS)
fprintf('Step 4: Running Collaborative Feature Selection (CFS)...\n');
fprintf('  Computing... (this may take 1-3 minutes)\n');

tic;
% Generate anchor points
X_anchor = generate_anchor_points(param.na, X_train);

% Run CFS algorithm
[projected_data_cfs, feature_subsets_cfs] = collaborative_feature_selection([X_test, X_train], ...
    X_train, X_anchor, y_train, num_divisions, param);

cfs_time = toc;
fprintf('✓ CFS completed! Time elapsed: %.2f seconds\n', cfs_time);

% Store CFS results
results.cfs.computation_time = cfs_time;
results.cfs.feature_subsets = feature_subsets_cfs;
results.cfs.projected_data = projected_data_cfs;

%% 6. Run Local Feature Selection (LFS)
fprintf('Step 5: Running Local Feature Selection (LFS) for comparison...\n');
fprintf('  Computing... (this may take 30-60 seconds)\n');

tic;
% Run LFS algorithm (using single division)
[projected_data_lfs, feature_subsets_lfs] = collaborative_feature_selection([X_test, X_train], ...
    X_train, X_anchor, y_train, 1, param);  % Single division = local

lfs_time = toc;
fprintf('✓ LFS completed! Time elapsed: %.2f seconds\n', lfs_time);

% Store LFS results
results.lfs.computation_time = lfs_time;
results.lfs.feature_subsets = feature_subsets_lfs;
results.lfs.projected_data = projected_data_lfs;

%% 7. Evaluate Performance
fprintf('Step 6: Evaluating algorithm performance...\n');

% Evaluate CFS performance
if ~isempty(feature_subsets_cfs)
    num_subsets_cfs = min(50, length(feature_subsets_cfs));
    cfs_nmi = zeros(num_subsets_cfs, 1);
    cfs_acc = zeros(num_subsets_cfs, 1);

    combined_labels = [y_test, y_train];
    fprintf('  Evaluating CFS results... ');

    for i = 1:num_subsets_cfs
        try
            cfs_nmi(i) = nmi(combined_labels, feature_subsets_cfs{i}');
            cfs_acc(i) = AccMeasure(combined_labels, feature_subsets_cfs{i});
        catch
            cfs_nmi(i) = NaN;
            cfs_acc(i) = NaN;
        end

        if mod(i, 10) == 0
            fprintf('%.0f%% ', (i/num_subsets_cfs)*100);
        end
    end
    fprintf('\n');

    results.cfs.nmi = cfs_nmi;
    results.cfs.acc = cfs_acc;
end

% Evaluate LFS performance
if ~isempty(feature_subsets_lfs)
    num_subsets_lfs = min(50, length(feature_subsets_lfs));
    lfs_nmi = zeros(num_subsets_lfs, 1);
    lfs_acc = zeros(num_subsets_lfs, 1);

    fprintf('  Evaluating LFS results... ');

    for i = 1:num_subsets_lfs
        try
            lfs_nmi(i) = nmi(combined_labels, feature_subsets_lfs{i}');
            lfs_acc(i) = AccMeasure(combined_labels, feature_subsets_lfs{i});
        catch
            lfs_nmi(i) = NaN;
            lfs_acc(i) = NaN;
        end

        if mod(i, 10) == 0
            fprintf('%.0f%% ', (i/num_subsets_lfs)*100);
        end
    end
    fprintf('\n');

    results.lfs.nmi = lfs_nmi;
    results.lfs.acc = lfs_acc;
end

fprintf('✓ Performance evaluation completed\n\n');

%% 8. Generate Comparison Figures
fprintf('Step 7: Generating comparison figures...\n');

% Figure 1: Comprehensive Performance Dashboard
figure('Position', [50, 50, 1400, 800], 'Name', [dataset_name ' - DCFS Performance Dashboard']);

% Calculate feature counts for x-axis (more meaningful than subset index)
if exist('cfs_acc', 'var') && ~isempty(cfs_acc)
    max_features = size(X_train, 1);
    feature_counts = round(linspace(max_features/length(cfs_acc), max_features, length(cfs_acc)));
end

% Subplot 1: Accuracy Comparison with clear context
subplot(2, 3, 1);
if exist('cfs_acc', 'var') && exist('lfs_acc', 'var')
    plot(feature_counts, cfs_acc, 'b-o', 'LineWidth', 3, 'MarkerSize', 6, 'DisplayName', 'CFS (Collaborative)');
    hold on;
    plot(feature_counts, lfs_acc, 'r--s', 'LineWidth', 3, 'MarkerSize', 6, 'DisplayName', 'LFS (Local Only)');

    % Add reference lines
    baseline_acc = mean([cfs_acc(1), lfs_acc(1)]);  % Starting performance
    yline(baseline_acc, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Baseline');

    xlabel('Number of Selected Features', 'FontSize', 12);
    ylabel('Classification Accuracy (%)', 'FontSize', 12);
    title('🎯 Algorithm Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 11);

    % Add performance annotations
    [max_cfs, idx_cfs] = max(cfs_acc);
    [max_lfs, idx_lfs] = max(lfs_acc);
    text(feature_counts(idx_cfs), max_cfs + 1, sprintf('Best: %.1f%%', max_cfs), ...
         'HorizontalAlignment', 'center', 'Color', 'blue', 'FontWeight', 'bold');
    text(feature_counts(idx_lfs), max_lfs - 2, sprintf('Best: %.1f%%', max_lfs), ...
         'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold');
end

% Subplot 2: Performance Gain Analysis
subplot(2, 3, 2);
if exist('cfs_acc', 'var') && exist('lfs_acc', 'var')
    gain = cfs_acc - lfs_acc;
    colors = gain;
    colors(gain >= 0) = 1; % Positive gains in blue
    colors(gain < 0) = 0;  % Negative gains in red

    bar(1:length(gain), gain, 'FaceColor', 'flat', 'CData', [colors', zeros(length(colors), 1), 1-colors']);
    xlabel('Feature Selection Stage', 'FontSize', 12);
    ylabel('CFS Advantage (%)', 'FontSize', 12);
    title('📊 CFS Performance Advantage', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    yline(0, 'k-', 'LineWidth', 2);
    set(gca, 'FontSize', 11);

    % Add summary text
    avg_gain = mean(gain);
    text(length(gain)/2, max(gain)*0.8, sprintf('Avg Advantage: %.1f%%', avg_gain), ...
         'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end

% Subplot 3: Algorithm Runtime Comparison
subplot(2, 3, 3);
if exist('cfs_time', 'var') && exist('lfs_time', 'var')
    runtime_data = [cfs_time, lfs_time];
    runtime_labels = {'CFS\n(Collaborative)', 'LFS\n(Local Only)'};
    colors = [0.2 0.4 0.8; 0.8 0.2 0.2];

    b = bar(runtime_data, 'FaceColor', 'flat');
    b.CData = colors;
    set(gca, 'XTickLabel', runtime_labels);
    ylabel('Runtime (seconds)', 'FontSize', 12);
    title('⏱️ Computational Cost', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 11);

    % Add value labels on bars
    for i = 1:length(runtime_data)
        text(i, runtime_data(i) + max(runtime_data)*0.02, ...
             sprintf('%.1fs', runtime_data(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

% Subplot 4: Feature Selection Effectiveness
subplot(2, 3, 4);
if exist('cfs_acc', 'var')
    improvement = cfs_acc - cfs_acc(1);  % Improvement from initial
    plot(1:length(improvement), improvement, 'g-o', 'LineWidth', 3, 'MarkerSize', 8);
    xlabel('Feature Selection Iteration', 'FontSize', 12);
    ylabel('Accuracy Improvement (%)', 'FontSize', 12);
    title('📈 Learning Progress', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 11);

    % Highlight best improvement
    [max_imp, best_iter] = max(improvement);
    text(best_iter, max_imp + 0.5, sprintf('Peak: +%.1f%%', max_imp), ...
         'HorizontalAlignment', 'center', 'Color', 'green', 'FontWeight', 'bold');
end

% Subplot 5: Summary Statistics
subplot(2, 3, 5);
axis off;
if exist('cfs_acc', 'var') && exist('lfs_acc', 'var')
    summary_text = {
        '📋 EXPERIMENT SUMMARY';
        '';
        sprintf('Dataset: %s', dataset_name);
        sprintf('Features: %d', size(X_train, 1));
        sprintf('Samples: %d', size(X_train, 2) + size(X_test, 2));
        '';
        '🏆 BEST RESULTS:';
        sprintf('CFS Best: %.1f%%', max(cfs_acc));
        sprintf('LFS Best: %.1f%%', max(lfs_acc));
        sprintf('Improvement: %.1f%%', max(cfs_acc) - max(lfs_acc));
        '';
        '⚡ EFFICIENCY:';
        sprintf('CFS Runtime: %.1fs', cfs_time);
        sprintf('LFS Runtime: %.1fs', lfs_time);
    };

    text(0.1, 0.9, summary_text, 'FontSize', 11, 'FontWeight', 'normal', ...
         'VerticalAlignment', 'top', 'BackgroundColor', [0.95 0.95 0.95], ...
         'EdgeColor', 'black', 'Margin', 10);
end

% Subplot 6: Algorithm Comparison Radar/Overview
subplot(2, 3, 6);
if exist('cfs_acc', 'var') && exist('lfs_acc', 'var')
    % Simple comparison chart
    metrics = {'Best\nAccuracy', 'Avg\nAccuracy', 'Consistency', 'Speed'};
    cfs_scores = [max(cfs_acc)/100, mean(cfs_acc)/100, 1-std(cfs_acc)/mean(cfs_acc), lfs_time/cfs_time];
    lfs_scores = [max(lfs_acc)/100, mean(lfs_acc)/100, 1-std(lfs_acc)/mean(lfs_acc), 1.0];

    x = 1:length(metrics);
    width = 0.35;

    bar(x - width/2, cfs_scores, width, 'FaceColor', [0.2 0.4 0.8], 'DisplayName', 'CFS');
    hold on;
    bar(x + width/2, lfs_scores, width, 'FaceColor', [0.8 0.2 0.2], 'DisplayName', 'LFS');

    set(gca, 'XTick', x, 'XTickLabel', metrics);
    ylabel('Normalized Score', 'FontSize', 12);
    title('⚖️ Overall Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    legend('FontSize', 10);
    grid on;
    set(gca, 'FontSize', 10);
end

% Add main title
sgtitle([dataset_name ' Dataset: Collaborative vs Local Feature Selection'], ...
        'FontSize', 16, 'FontWeight', 'bold');

% Save comprehensive figure with simple names
saveas(gcf, ['./figures/' dataset_name '_Results.fig']);
saveas(gcf, ['./figures/' dataset_name '_Results.png']);

fprintf('✓ Results dashboard saved\n');

%% 9. Generate Feature Weight Visualization
if exist('feature_subsets_cfs', 'var') && ~isempty(feature_subsets_cfs)
    figure('Position', [150, 150, 1000, 600], 'Name', [dataset_name ' - Feature Weights']);

    % Show first 20 features for visualization
    n_show = min(20, size(X_train, 1));

    if length(feature_subsets_cfs) >= 10
        % Use 10th subset as example
        subset_to_show = feature_subsets_cfs{10};
        feature_weights = abs(subset_to_show(1:n_show));

        bar(1:n_show, feature_weights);
        xlabel('Feature Index');
        ylabel('Feature Weight');
        title([dataset_name ' - Feature Weights Distribution (Subset 10)']);
        grid on;
        set(gca, 'FontSize', 12);

        % Save figure with simple name
        saveas(gcf, ['./figures/' dataset_name '_Weights.fig']);
        saveas(gcf, ['./figures/' dataset_name '_Weights.png']);

        fprintf('✓ Feature weights figure saved\n');
    end
end

%% 10. Save Results
fprintf('Step 8: Saving experiment results...\n');

% Save results to .mat file with simple name
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
results_filename = sprintf('./figures/%s_Data_%s.mat', dataset_name, timestamp);
save(results_filename, 'results');

% Generate summary report with simple name
report_filename = sprintf('./figures/%s_Report_%s.txt', dataset_name, timestamp);
fid = fopen(report_filename, 'w');
fprintf(fid, '%s Dataset - DCFS Experiment Report\n', dataset_name);
fprintf(fid, '=====================================\n\n');
fprintf(fid, 'Experiment Date: %s\n', datestr(now));
fprintf(fid, 'Dataset: %s\n', dataset_name);
fprintf(fid, 'Training samples: %d\n', size(X_train, 2));
fprintf(fid, 'Test samples: %d\n', size(X_test, 2));
fprintf(fid, 'Features: %d\n', size(X_train, 1));
fprintf(fid, 'Classes: %d\n', length(unique(y_train)));
fprintf(fid, '\nAlgorithm Performance:\n');
fprintf(fid, 'CFS Runtime: %.2f seconds\n', cfs_time);
fprintf(fid, 'LFS Runtime: %.2f seconds\n', lfs_time);

if exist('cfs_nmi', 'var') && exist('cfs_acc', 'var')
    fprintf(fid, '\nCFS Results:\n');
    fprintf(fid, 'Best NMI: %.3f\n', max(cfs_nmi(~isnan(cfs_nmi))));
    fprintf(fid, 'Best Accuracy: %.1f%%\n', max(cfs_acc(~isnan(cfs_acc))));
end

if exist('lfs_nmi', 'var') && exist('lfs_acc', 'var')
    fprintf(fid, '\nLFS Results:\n');
    fprintf(fid, 'Best NMI: %.3f\n', max(lfs_nmi(~isnan(lfs_nmi))));
    fprintf(fid, 'Best Accuracy: %.1f%%\n', max(lfs_acc(~isnan(lfs_acc))));
end

fclose(fid);

fprintf('✓ Results saved to: %s\n', results_filename);
fprintf('✓ Report saved to: %s\n', report_filename);

%% 11. Summary
fprintf('\n=== Demo Summary ===\n');
fprintf('✓ Dataset: %s\n', dataset_name);
fprintf('✓ CFS algorithm runtime: %.2f seconds\n', cfs_time);
fprintf('✓ LFS algorithm runtime: %.2f seconds\n', lfs_time);
fprintf('✓ Figures saved to ./figures/ directory\n');

if exist('cfs_acc', 'var') && exist('lfs_acc', 'var')
    cfs_best = max(cfs_acc(~isnan(cfs_acc)));
    lfs_best = max(lfs_acc(~isnan(lfs_acc)));
    fprintf('✓ Best CFS accuracy: %.1f%%\n', cfs_best);
    fprintf('✓ Best LFS accuracy: %.1f%%\n', lfs_best);

    if cfs_best > lfs_best
        fprintf('🏆 CFS outperformed LFS by %.1f percentage points\n', cfs_best - lfs_best);
    else
        fprintf('🏆 LFS outperformed CFS by %.1f percentage points\n', lfs_best - cfs_best);
    end
end

fprintf('\n💡 Next Steps:\n');
fprintf('   - Check ./figures/ for generated plots\n');
fprintf('   - Compare with original .fig files in ./experiments/\n');
fprintf('   - Try different parameter settings\n');
fprintf('   - Run with the other dataset\n\n');

fprintf('🎉 Demo completed successfully!\n');