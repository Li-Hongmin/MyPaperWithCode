function param = get_default_params(dataset_type, performance_mode)
%GET_DEFAULT_PARAMS Get default parameters for DCFS algorithms
%
% Input:
%   dataset_type - 'leukemia', 'mnist', 'general' (default: 'general')
%   performance_mode - 'fast', 'balanced', 'accurate' (default: 'balanced')
%
% Output:
%   param - parameter structure with optimized defaults
%
% Examples:
%   param = get_default_params();                    % General balanced
%   param = get_default_params('leukemia', 'fast');  % Fast leukemia
%   param = get_default_params('mnist', 'accurate'); % Accurate MNIST

if nargin < 1
    dataset_type = 'general';
end

if nargin < 2
    performance_mode = 'balanced';
end

%% Base parameters (common to all configurations)
param.k = 3;              % Number of classes (will be auto-detected if possible)
param.delta1 = 5e-3;      % Regularization parameter for ME methods
param.delta2 = 2;         % Regularization parameter for ME methods
param.delta3 = 1.35;      % Range of eigenvalue for ME methods
param.kernel = 'L';       % Kernel type: 'L' (Linear) or 'G' (Gaussian)
param.ksigma = 0;         % Sigma for Gaussian kernel (0 = auto-tuning)

%% Dataset-specific optimizations
switch lower(dataset_type)
    case 'leukemia'
        % Optimized for gene expression data (high-dimensional, few samples)
        switch lower(performance_mode)
            case 'fast'
                param.na = 50;    % Number of anchor points
                param.neig = 8;   % Number of eigenvalues
            case 'balanced'
                param.na = 100;   % Number of anchor points
                param.neig = 12;  % Number of eigenvalues
            case 'accurate'
                param.na = 200;   % Number of anchor points
                param.neig = 18;  % Number of eigenvalues
        end

    case 'mnist'
        % Optimized for image data (moderate dimensions, more samples)
        switch lower(performance_mode)
            case 'fast'
                param.na = 200;   % Number of anchor points
                param.neig = 50;  % Number of eigenvalues
                param.delta3 = 0.8;
            case 'balanced'
                param.na = 500;   % Number of anchor points
                param.neig = 100; % Number of eigenvalues
                param.delta3 = 0.8;
            case 'accurate'
                param.na = 1000;  % Number of anchor points
                param.neig = 160; % Number of eigenvalues
                param.delta3 = 0.8;
        end
        param.k = 10;  % MNIST has 10 classes

    case '2d'
        % Optimized for 2D synthetic data
        param.na = 100;
        param.neig = 50;
        param.delta3 = 0.7;

    otherwise  % 'general' or unknown
        % Conservative defaults for unknown data types
        switch lower(performance_mode)
            case 'fast'
                param.na = 50;    % Number of anchor points
                param.neig = 10;  % Number of eigenvalues
            case 'balanced'
                param.na = 100;   % Number of anchor points
                param.neig = 15;  % Number of eigenvalues
            case 'accurate'
                param.na = 300;   % Number of anchor points
                param.neig = 25;  % Number of eigenvalues
        end
end

%% Performance-specific adjustments
switch lower(performance_mode)
    case 'fast'
        % Optimize for speed
        param.max_iter = 5;       % Reduce iterations
        param.convergence_tol = 1e-4;  % Looser convergence

    case 'balanced'
        % Balance speed and accuracy
        param.max_iter = 10;      % Standard iterations
        param.convergence_tol = 1e-6;  % Standard convergence

    case 'accurate'
        % Optimize for accuracy
        param.max_iter = 15;      % More iterations
        param.convergence_tol = 1e-8;  % Tighter convergence
end

%% Validation
param = validate_params(param);

fprintf('Parameters configured for %s dataset in %s mode\n', dataset_type, performance_mode);
fprintf('  Anchors: %d, Eigenvalues: %d, Max iterations: %d\n', ...
    param.na, param.neig, param.max_iter);

end

function param = validate_params(param)
%VALIDATE_PARAMS Validate and adjust parameters if necessary

% Ensure positive values
param.na = max(1, param.na);
param.neig = max(1, param.neig);
param.k = max(1, param.k);

% Ensure reasonable ranges
param.delta1 = max(1e-8, param.delta1);
param.delta2 = max(0.1, param.delta2);
param.delta3 = max(0.1, param.delta3);

% Warn about potential issues
if param.na > 1000
    warning('Large number of anchor points (%d) may cause memory issues', param.na);
end

if param.neig > 200
    warning('Large number of eigenvalues (%d) may slow computation', param.neig);
end

end