function [training_data, classification_results] = evaluate_feature_subsets(X_test, division_data, feature_ranking, algorithm_params)
%EVALUATE_FEATURE_SUBSETS Evaluate classification performance for different feature subsets
% OPTIMIZED VERSION: Precomputes kernel matrices to avoid repeated calculations
%
% Input:
%   X_test           - Test data matrix
%   division_data    - Structure array with optimized parameters
%   feature_ranking  - Feature ranking indices
%   algorithm_params - Parameter structure
%
% Output:
%   training_data         - Feature selection results (cell array)
%   classification_results - Classification results for different feature subsets

num_divisions = length(division_data);
test_division_idx = randperm(num_divisions, 1);  % Use same division as ranking
num_features = length(feature_ranking);

% Get full training and test data
X_train_full = division_data(test_division_idx).X;
X_test_full = X_test;
y_train_full = division_data(test_division_idx).l;

% Initialize output cell arrays
classification_results = cell(num_features, 1);

% OPTIMIZATION: Direct computation without calling kernel_least_squares_classification in loop
fprintf('  Optimizing feature subset evaluation...\n');

% Prepare label matrix for classification
num_train_samples = size(X_train_full, 2);
if algorithm_params.k == 2
    num_output_vars = 1;
    label_matrix = -(y_train_full-1.5)*2;
else
    num_output_vars = algorithm_params.k;
    label_matrix = zeros(algorithm_params.k, num_train_samples);
    for class_idx = 1:algorithm_params.k
        label_matrix(class_idx, y_train_full==class_idx) = 1;
        label_matrix(class_idx, y_train_full~=class_idx) = 0;
    end
end

% OPTIMIZATION: Evaluate each feature subset using optimized computation
regularization_param = sqrt(algorithm_params.delta1);

for i = 1:num_features
    selected_features = feature_ranking(1:i);

    % OPTIMIZED: Direct computation without redundant kernel calls
    X_train_subset = X_train_full(selected_features, :);
    X_test_subset = X_test_full(selected_features, :);

    % Compute kernel matrices for current subset
    K_subset = X_train_subset' * X_train_subset;
    K2_subset = X_train_subset' * X_test_subset;

    % Solve classification problem using backslash for better performance
    K_regularized = K_subset + regularization_param * eye(size(K_subset, 1));
    classification_weights = K_regularized \ (label_matrix');

    % Predict labels
    predictions = K2_subset' * classification_weights;
    if num_output_vars == 1
        classification_results{i} = (predictions < 0) + 1;
    else
        [~, classification_results{i}] = max(predictions');
    end
end

fprintf('  ✓ Optimized evaluation completed\n');

% Return selected training data and labels
training_data{1} = division_data(test_division_idx).X;
training_data{2} = division_data(test_division_idx).l;

end