%% CFSnmiAcc Usage Example
% Demonstrates how to generate and use CFS algorithm performance matrix

clear; clc; close all;

fprintf('=== CFSnmiAcc Usage Example ===\n\n');

%% Method 1: Generate new CFSnmiAcc
fprintf('Method 1: Generating new CFSnmiAcc matrix...\n');
% CFSnmiAcc = generate_CFSnmiAcc();  % Full run (requires 1-2 minutes)

%% Method 2: Load existing CFSnmiAcc
fprintf('Method 2: Loading existing CFSnmiAcc...\n');

% Find latest CFSnmiAcc file
files = dir('CFSnmiAcc_*.mat');
if ~isempty(files)
    % Sort by time, select latest
    [~, idx] = max([files.datenum]);
    latest_file = files(idx).name;

    fprintf('Loading file: %s\n', latest_file);
    load(latest_file);

    fprintf('✓ CFSnmiAcc loaded successfully, size: %d × %d\n', size(CFSnmiAcc));
else
    error('CFSnmiAcc file not found, please run generate_CFSnmiAcc() first');
end

%% Data Analysis
fprintf('\n=== Data Analysis ===\n');

% Basic statistics
fprintf('Data Statistics:\n');
fprintf('  Total subsets: %d\n', size(CFSnmiAcc, 1));
fprintf('  NMI range: [%.3f, %.3f]\n', min(CFSnmiAcc(:,1)), max(CFSnmiAcc(:,1)));
fprintf('  Accuracy range: [%.1f%%, %.1f%%]\n', min(CFSnmiAcc(:,2)), max(CFSnmiAcc(:,2)));
fprintf('  Average NMI: %.3f (±%.3f)\n', mean(CFSnmiAcc(:,1)), std(CFSnmiAcc(:,1)));
fprintf('  Average accuracy: %.1f%% (±%.1f%%)\n', mean(CFSnmiAcc(:,2)), std(CFSnmiAcc(:,2)));

% Find best performance points
[max_nmi, idx_nmi] = max(CFSnmiAcc(:,1));
[max_acc, idx_acc] = max(CFSnmiAcc(:,2));

fprintf('\nBest Performance:\n');
fprintf('  Highest NMI: %.3f (feature subset #%d)\n', max_nmi, idx_nmi);
fprintf('  Highest accuracy: %.1f%% (feature subset #%d)\n', max_acc, idx_acc);

%% Performance Trend Analysis
fprintf('\n=== Performance Trend Analysis ===\n');

% Analyze trend of first 100 subsets
n_analyze = min(100, size(CFSnmiAcc, 1));
early_subsets = CFSnmiAcc(1:n_analyze, :);

fprintf('Analysis of first %d feature subsets:\n', n_analyze);
fprintf('  NMI improvement: %.3f → %.3f (%.1f%% increase)\n', ...
    early_subsets(1,1), early_subsets(end,1), ...
    (early_subsets(end,1)/early_subsets(1,1)-1)*100);

fprintf('  Accuracy improvement: %.1f%% → %.1f%% (%.1f percentage points)\n', ...
    early_subsets(1,2), early_subsets(end,2), ...
    early_subsets(end,2) - early_subsets(1,2));

%% Visualization Analysis (if GUI supported)
fprintf('\n=== Visualization Analysis ===\n');

try
    % Create figure
    figure('Position', [100, 100, 1200, 400], 'Name', 'CFSnmiAcc Performance Analysis');

    % Subplot 1: NMI trend
    subplot(1, 3, 1);
    plot(CFSnmiAcc(:,1), 'b-', 'LineWidth', 1.5);
    xlabel('Feature Subset Index');
    ylabel('NMI');
    title('NMI Performance Trend');
    grid on;

    % Mark best point
    hold on;
    plot(idx_nmi, max_nmi, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(idx_nmi, max_nmi, sprintf(' Best: %.3f', max_nmi), 'FontSize', 10);

    % Subplot 2: Accuracy trend
    subplot(1, 3, 2);
    plot(CFSnmiAcc(:,2), 'r-', 'LineWidth', 1.5);
    xlabel('Feature Subset Index');
    ylabel('Accuracy (%)');
    title('Accuracy Performance Trend');
    grid on;

    % Mark best point
    hold on;
    plot(idx_acc, max_acc, 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    text(idx_acc, max_acc, sprintf(' Best: %.1f%%', max_acc), 'FontSize', 10);

    % Subplot 3: NMI vs Accuracy scatter plot
    subplot(1, 3, 3);
    scatter(CFSnmiAcc(:,1), CFSnmiAcc(:,2), 20, 1:size(CFSnmiAcc,1), 'filled');
    xlabel('NMI');
    ylabel('Accuracy (%)');
    title('NMI vs Accuracy Relationship');
    colorbar;
    colormap('jet');
    grid on;

    % Mark best points
    hold on;
    plot(max_nmi, CFSnmiAcc(idx_nmi,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
    plot(CFSnmiAcc(idx_acc,1), max_acc, 'k^', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Data Points', 'Best NMI', 'Best Accuracy', 'Location', 'best');

    fprintf('✓ Visualization charts generated\n');

catch ME
    fprintf('⚠️ Visualization failed (possibly no GUI environment): %s\n', ME.message);
end

%% Export Analysis Report
fprintf('\n=== Export Analysis Report ===\n');

report_filename = sprintf('CFSnmiAcc_analysis_report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(report_filename, 'w');

fprintf(fid, 'CFS Algorithm Performance Analysis Report\n');
fprintf(fid, '===========================================\n\n');
fprintf(fid, 'Analysis Time: %s\n', datestr(now));
fprintf(fid, 'Data Source: %s\n', latest_file);
fprintf(fid, '\nBasic Statistics:\n');
fprintf(fid, 'Total Feature Subsets: %d\n', size(CFSnmiAcc, 1));
fprintf(fid, 'NMI Mean: %.3f (Std: %.3f)\n', mean(CFSnmiAcc(:,1)), std(CFSnmiAcc(:,1)));
fprintf(fid, 'Accuracy Mean: %.1f%% (Std: %.1f%%)\n', mean(CFSnmiAcc(:,2)), std(CFSnmiAcc(:,2)));
fprintf(fid, '\nBest Performance:\n');
fprintf(fid, 'Highest NMI: %.3f (subset #%d)\n', max_nmi, idx_nmi);
fprintf(fid, 'Highest Accuracy: %.1f%% (subset #%d)\n', max_acc, idx_acc);
fprintf(fid, '\nPerformance Range:\n');
fprintf(fid, 'NMI: [%.3f, %.3f] (variation: %.1f%%)\n', ...
    min(CFSnmiAcc(:,1)), max(CFSnmiAcc(:,1)), ...
    (max(CFSnmiAcc(:,1))/min(CFSnmiAcc(:,1))-1)*100);
fprintf(fid, 'Accuracy: [%.1f%%, %.1f%%] (variation: %.1f percentage points)\n', ...
    min(CFSnmiAcc(:,2)), max(CFSnmiAcc(:,2)), ...
    max(CFSnmiAcc(:,2)) - min(CFSnmiAcc(:,2)));

fclose(fid);
fprintf('✓ Analysis report saved to: %s\n', report_filename);

%% Usage Recommendations
fprintf('\n=== Usage Recommendations ===\n');
fprintf('1. Best feature subset recommendations:\n');
fprintf('   - If prioritizing NMI: use subset #%d (NMI=%.3f, ACC=%.1f%%)\n', ...
    idx_nmi, max_nmi, CFSnmiAcc(idx_nmi,2));
fprintf('   - If prioritizing accuracy: use subset #%d (ACC=%.1f%%, NMI=%.3f)\n', ...
    idx_acc, max_acc, CFSnmiAcc(idx_acc,1));

% Find balanced point (high NMI and accuracy)
balanced_score = 0.6 * (CFSnmiAcc(:,1)/max(CFSnmiAcc(:,1))) + ...
                 0.4 * (CFSnmiAcc(:,2)/max(CFSnmiAcc(:,2)));
[~, idx_balanced] = max(balanced_score);

fprintf('   - If balancing both: use subset #%d (NMI=%.3f, ACC=%.1f%%)\n', ...
    idx_balanced, CFSnmiAcc(idx_balanced,1), CFSnmiAcc(idx_balanced,2));

fprintf('\n2. Data usage tips:\n');
fprintf('   - CFSnmiAcc(:,1) = NMI values\n');
fprintf('   - CFSnmiAcc(:,2) = Accuracy (%)\n');
fprintf('   - Row index = Feature subset number\n');

fprintf('\n🎉 CFSnmiAcc analysis completed!\n');