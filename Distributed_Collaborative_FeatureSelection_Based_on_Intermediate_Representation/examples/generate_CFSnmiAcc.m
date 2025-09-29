function CFSnmiAcc = generate_CFSnmiAcc(save_to_file)
%GENERATE_CFSNMIACC Generate performance matrix of NMI and accuracy for CFS algorithm
%
% Input:
%   save_to_file - (optional) Whether to save to file, default true
%
% Output:
%   CFSnmiAcc - N×2 matrix, 1st column: NMI, 2nd column: accuracy (%)
%
% Examples:
%   CFSnmiAcc = generate_CFSnmiAcc();           % Generate and save
%   CFSnmiAcc = generate_CFSnmiAcc(false);     % Generate only, no save

if nargin < 1
    save_to_file = true;
end

fprintf('=== Generate CFS Algorithm Performance Matrix (CFSnmiAcc) ===\n\n');

%% 1. Load Data
fprintf('Step 1: Loading Leukemia dataset...\n');
try
    load('./DATA_SET/leukemia/leukemia_data.mat');
    load('./DATA_SET/leukemia/leukemia_label.mat');
    fprintf('✓ Data loaded successfully\n');
catch ME
    error('❌ Data files not found: %s', ME.message);
end

% Data preprocessing
X_train = data.train';
X_test = data.test';
y_train = label.train';
y_test = label.test';

fprintf('  Data dimensions: %d features × %d train samples × %d test samples\n', ...
    size(X_train,1), size(X_train,2), size(X_test,2));

%% 2. Set Parameters
fprintf('\nStep 2: Setting CFS algorithm parameters...\n');
param.k = 3;              % Number of classes
param.delta1 = 5e-3;      % Regularization parameter
param.delta2 = 2;
param.delta3 = 1.35;
param.neig = 12;          % Number of eigenvalues
param.na = 30;            % Number of anchors (must be less than 38 training samples)
param.kernel = 'L';       % Linear kernel
param.ksigma = 0;

nd = 2;  % Number of distributed nodes

fprintf('  Number of anchors: %d\n', param.na);
fprintf('  Number of eigenvalues: %d\n', param.neig);

%% 3. Run CFS Algorithm
fprintf('\nStep 3: Running CFS distributed collaborative feature selection...\n');
fprintf('  Computing... (estimated 1-2 minutes)\n');

tic;
% Generate anchor points
X_anchor = generate_anchor_points(param.na, X_train);

% Run CFS algorithm
[projected_data, feature_subsets] = collaborative_feature_selection([X_test, X_train], ...
    X_train, X_anchor, y_train, nd, param);

computation_time = toc;
fprintf('✓ CFS algorithm completed! Time elapsed: %.2f seconds\n', computation_time);

%% 4. Generate Performance Matrix
if isempty(feature_subsets)
    error('❌ CFS algorithm failed to generate feature subsets');
end

fprintf('\nStep 4: Computing NMI and accuracy for each feature subset...\n');
n_subsets = length(feature_subsets);
fprintf('  Total feature subsets: %d\n', n_subsets);

% Initialize result matrix
CFSnmiAcc = zeros(n_subsets, 2);
combined_labels = [y_test, y_train];  % Combine test and training labels

% Compute performance for each subset
success_count = 0;
fprintf('  Computing progress: ');

for i = 1:n_subsets
    try
        % Calculate NMI (Normalized Mutual Information)
        nmi_val = nmi(combined_labels, feature_subsets{i}');

        % Calculate accuracy
        acc_val = AccMeasure(combined_labels, feature_subsets{i});

        % Store results
        CFSnmiAcc(i, 1) = nmi_val;
        CFSnmiAcc(i, 2) = acc_val;

        success_count = success_count + 1;

        % Display progress
        if mod(i, max(1, floor(n_subsets/20))) == 0
            fprintf('%.0f%% ', (i/n_subsets)*100);
        end

    catch ME
        % If calculation fails for a subset, set as NaN
        CFSnmiAcc(i, 1) = NaN;
        CFSnmiAcc(i, 2) = NaN;

        if i <= 10  % Only show first 10 errors
            fprintf('\n  Warning: Subset %d calculation failed (%s)\n', i, ME.message);
        end
    end
end

fprintf('\n✓ Performance calculation completed! Success rate: %.1f%% (%d/%d)\n', ...
    success_count/n_subsets*100, success_count, n_subsets);

%% 5. Data Statistics and Cleaning
fprintf('\nStep 5: Data statistics and cleaning...\n');

% Remove NaN values
valid_rows = ~any(isnan(CFSnmiAcc), 2);
CFSnmiAcc_clean = CFSnmiAcc(valid_rows, :);

fprintf('  Original data: %d × 2\n', size(CFSnmiAcc, 1));
fprintf('  Cleaned data: %d × 2\n', size(CFSnmiAcc_clean, 1));

if ~isempty(CFSnmiAcc_clean)
    fprintf('  NMI range: [%.3f, %.3f]\n', ...
        min(CFSnmiAcc_clean(:,1)), max(CFSnmiAcc_clean(:,1)));
    fprintf('  Accuracy range: [%.1f%%, %.1f%%]\n', ...
        min(CFSnmiAcc_clean(:,2)), max(CFSnmiAcc_clean(:,2)));

    % Find best performance
    [max_nmi, idx_nmi] = max(CFSnmiAcc_clean(:,1));
    [max_acc, idx_acc] = max(CFSnmiAcc_clean(:,2));

    fprintf('  Best NMI: %.3f (subset #%d)\n', max_nmi, idx_nmi);
    fprintf('  Best accuracy: %.1f%% (subset #%d)\n', max_acc, idx_acc);
end

% Use cleaned data
CFSnmiAcc = CFSnmiAcc_clean;

%% 6. Save Results
if save_to_file && ~isempty(CFSnmiAcc)
    fprintf('\nStep 6: Saving results...\n');

    % Save as .mat file
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('CFSnmiAcc_%s.mat', timestamp);

    save(filename, 'CFSnmiAcc', 'param', 'computation_time', 'n_subsets');
    fprintf('✓ Results saved to: %s\n', filename);

    % Also save as CSV file for other software use
    csv_filename = sprintf('CFSnmiAcc_%s.csv', timestamp);
    csvwrite(csv_filename, CFSnmiAcc);
    fprintf('✓ CSV file saved to: %s\n', csv_filename);

    % Save summary information
    summary_filename = sprintf('CFSnmiAcc_summary_%s.txt', timestamp);
    fid = fopen(summary_filename, 'w');
    fprintf(fid, 'CFS Algorithm Performance Results Summary\n');
    fprintf(fid, '=====================\n\n');
    fprintf(fid, 'Generation time: %s\n', datestr(now));
    fprintf(fid, 'Dataset: Leukemia\n');
    fprintf(fid, 'Algorithm: Distributed Collaborative Feature Selection (CFS)\n');
    fprintf(fid, 'Runtime: %.2f seconds\n', computation_time);
    fprintf(fid, 'Number of anchors: %d\n', param.na);
    fprintf(fid, 'Number of eigenvalues: %d\n', param.neig);
    fprintf(fid, 'Distributed nodes: %d\n', nd);
    fprintf(fid, '\nPerformance statistics:\n');
    fprintf(fid, 'Total feature subsets: %d\n', n_subsets);
    fprintf(fid, 'Valid subsets: %d\n', size(CFSnmiAcc, 1));
    if ~isempty(CFSnmiAcc)
        fprintf(fid, 'Best NMI: %.3f\n', max(CFSnmiAcc(:,1)));
        fprintf(fid, 'Best accuracy: %.1f%%\n', max(CFSnmiAcc(:,2)));
        fprintf(fid, 'Average NMI: %.3f\n', mean(CFSnmiAcc(:,1)));
        fprintf(fid, 'Average accuracy: %.1f%%\n', mean(CFSnmiAcc(:,2)));
    end
    fclose(fid);
    fprintf('✓ Summary information saved to: %s\n', summary_filename);
end

%% 7. Display Results Preview
if ~isempty(CFSnmiAcc)
    fprintf('\n=== CFSnmiAcc Results Preview ===\n');
    fprintf('Matrix size: %d × 2 (NMI, Accuracy)\n', size(CFSnmiAcc, 1));
    fprintf('\nFirst 10 rows of results:\n');
    fprintf('  Subset    NMI     Accuracy(%%)\n');
    fprintf('  ------   -----   --------\n');

    preview_rows = min(10, size(CFSnmiAcc, 1));
    for i = 1:preview_rows
        fprintf('  %3d      %.3f    %6.1f\n', i, CFSnmiAcc(i,1), CFSnmiAcc(i,2));
    end

    if size(CFSnmiAcc, 1) > 10
        fprintf('  ...      ...      ...\n');
        fprintf('Total %d rows of data\n', size(CFSnmiAcc, 1));
    end
else
    fprintf('⚠️ Generated CFSnmiAcc is empty\n');
end

fprintf('\n🎉 CFSnmiAcc generation completed!\n');

end