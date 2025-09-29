function validate_inputs(X_test, X_train, X_anchor, labels_train, num_divisions, algorithm_params)
%VALIDATE_INPUTS Comprehensive input validation for DCFS algorithms
%
% Input:
%   X_test           - Test data matrix (features × samples)
%   X_train          - Training data matrix (features × samples)
%   X_anchor         - Anchor data matrix (features × samples)
%   labels_train     - Training labels vector
%   num_divisions    - Number of divisions
%   algorithm_params - Parameter structure
%
% Throws error if inputs are invalid, otherwise returns silently

%% Validate data matrices
if ~isnumeric(X_test) || ~isreal(X_test)
    error('validate_inputs:invalid_test_data', 'X_test must be a real numeric matrix');
end

if ~isnumeric(X_train) || ~isreal(X_train)
    error('validate_inputs:invalid_train_data', 'X_train must be a real numeric matrix');
end

if ~isnumeric(X_anchor) || ~isreal(X_anchor)
    error('validate_inputs:invalid_anchor_data', 'X_anchor must be a real numeric matrix');
end

%% Check matrix dimensions
[d_test, n_test] = size(X_test);
[d_train, n_train] = size(X_train);
[d_anchor, n_anchor] = size(X_anchor);

if d_test ~= d_train || d_train ~= d_anchor
    error('validate_inputs:dimension_mismatch', ...
        'All data matrices must have same number of features. Got: test=%d, train=%d, anchor=%d', ...
        d_test, d_train, d_anchor);
end

%% Validate labels
if ~isnumeric(labels_train) || ~isvector(labels_train)
    error('validate_inputs:invalid_labels', 'labels_train must be a numeric vector');
end

if length(labels_train) ~= n_train
    error('validate_inputs:label_size_mismatch', ...
        'Number of labels (%d) must match number of training samples (%d)', ...
        length(labels_train), n_train);
end

% Check for valid label values
unique_labels = unique(labels_train);
if any(unique_labels < 1) || any(mod(unique_labels, 1) ~= 0)
    error('validate_inputs:invalid_label_values', ...
        'Labels must be positive integers');
end

%% Validate number of divisions
if ~isnumeric(num_divisions) || ~isscalar(num_divisions) || num_divisions < 1 || mod(num_divisions, 1) ~= 0
    error('validate_inputs:invalid_divisions', ...
        'Number of divisions (num_divisions) must be a positive integer');
end

if num_divisions > n_train
    error('validate_inputs:too_many_divisions', ...
        'Number of divisions (%d) cannot exceed number of training samples (%d)', ...
        num_divisions, n_train);
end

if n_train / num_divisions < 2
    warning('validate_inputs:few_samples_per_division', ...
        'Very few samples per division (%.1f). Consider reducing num_divisions', n_train / num_divisions);
end

%% Validate parameter structure
if ~isstruct(algorithm_params)
    error('validate_inputs:invalid_param', 'algorithm_params must be a structure');
end

% Check required fields
required_fields = {'k', 'delta1', 'delta2', 'delta3', 'neig', 'na', 'kernel'};
for i = 1:length(required_fields)
    field = required_fields{i};
    if ~isfield(algorithm_params, field)
        error('validate_inputs:missing_field', ...
            'algorithm_params must contain field: %s', field);
    end
end

% Validate specific parameter values
if algorithm_params.k < 1 || mod(algorithm_params.k, 1) ~= 0
    error('validate_inputs:invalid_k', 'algorithm_params.k must be a positive integer');
end

if algorithm_params.neig < 1 || algorithm_params.neig > min(d_train, n_train)
    error('validate_inputs:invalid_neig', ...
        'algorithm_params.neig must be between 1 and min(features, samples) = %d', ...
        min(d_train, n_train));
end

if algorithm_params.na < 1 || algorithm_params.na > n_train
    error('validate_inputs:invalid_na', ...
        'algorithm_params.na must be between 1 and number of training samples (%d)', n_train);
end

if algorithm_params.delta1 <= 0 || algorithm_params.delta2 <= 0 || algorithm_params.delta3 <= 0
    error('validate_inputs:invalid_deltas', ...
        'All delta parameters must be positive');
end

if ~ischar(algorithm_params.kernel) || ~ismember(upper(algorithm_params.kernel), {'L', 'G'})
    error('validate_inputs:invalid_kernel', ...
        'algorithm_params.kernel must be ''L'' (linear) or ''G'' (gaussian)');
end

%% Additional data quality checks
% Check for NaN or Inf values
if any(~isfinite(X_test(:)))
    error('validate_inputs:invalid_test_values', ...
        'X_test contains NaN or Inf values');
end

if any(~isfinite(X_train(:)))
    error('validate_inputs:invalid_train_values', ...
        'X_train contains NaN or Inf values');
end

if any(~isfinite(X_anchor(:)))
    error('validate_inputs:invalid_anchor_values', ...
        'X_anchor contains NaN or Inf values');
end

% Check for constant features
train_var = var(X_train, 0, 2);
constant_features = sum(train_var < eps);
if constant_features > 0
    warning('validate_inputs:constant_features', ...
        'Found %d constant features (zero variance). Consider feature selection', ...
        constant_features);
end

% Check data scale
max_values = max(abs(X_train(:)));
if max_values > 1e6
    warning('validate_inputs:large_values', ...
        'Data contains very large values (max: %.2e). Consider normalization', ...
        max_values);
end

%% Success message (only in verbose mode)
if isfield(algorithm_params, 'verbose') && algorithm_params.verbose
    fprintf('✓ Input validation passed successfully\n');
    fprintf('  Features: %d, Train samples: %d, Test samples: %d, Anchors: %d\n', ...
        d_train, n_train, n_test, n_anchor);
    fprintf('  Divisions: %d, Classes: %d\n', num_divisions, length(unique_labels));
end

end