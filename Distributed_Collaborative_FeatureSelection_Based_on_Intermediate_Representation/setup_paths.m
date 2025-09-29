function setup_paths()
%SETUP_PATHS Add all necessary paths for DCFS project
%
% This function adds the core, utils, tests, and examples directories to
% MATLAB path to ensure all functions are accessible.

% Get the current script's directory (project root)
current_dir = fileparts(mfilename('fullpath'));

% Add subdirectories to path
addpath(fullfile(current_dir, 'core'));
addpath(fullfile(current_dir, 'utils'));
addpath(fullfile(current_dir, 'tests'));
addpath(fullfile(current_dir, 'examples'));
addpath(fullfile(current_dir, 'DATA_SET'));

% Add subdirectories of DATA_SET
if exist(fullfile(current_dir, 'DATA_SET', 'leukemia'), 'dir')
    addpath(fullfile(current_dir, 'DATA_SET', 'leukemia'));
end

if exist(fullfile(current_dir, 'DATA_SET', 'MNIST'), 'dir')
    addpath(fullfile(current_dir, 'DATA_SET', 'MNIST'));
end

fprintf('✅ DCFS paths configured successfully!\n');
fprintf('   Core algorithms: %s\n', fullfile(current_dir, 'core'));
fprintf('   Utility functions: %s\n', fullfile(current_dir, 'utils'));
fprintf('   Test scripts: %s\n', fullfile(current_dir, 'tests'));
fprintf('   Examples: %s\n', fullfile(current_dir, 'examples'));

end