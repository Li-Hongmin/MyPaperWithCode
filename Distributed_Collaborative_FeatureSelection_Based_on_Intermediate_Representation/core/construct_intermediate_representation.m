function [Div_data, intermediate_dims] = construct_intermediate_representation(Div_data, X_test, X_anchor, algorithm_params)
%CONSTRUCT_INTERMEDIATE_REPRESENTATION Build intermediate representations for each division
%
% Input:
%   Div_data - Structure array containing partitioned data
%   X_test    - Test data matrix
%   X_anchor     - Anchor data matrix
%   algorithm_params    - Parameter structure
%
% Output:
%   Div_data - Updated structure array with intermediate representations
%   intermediate_dims     - Number of dimensions in intermediate representation

num_divisions = length(Div_data);
num_test_samples = size(X_test, 2);

for division_idx = 1:num_divisions
    [tmp, Div_data(division_idx).Y] = kernel_locality_preserving_projection([X_test, X_anchor], Div_data(division_idx).X, algorithm_params);

    Div_data(division_idx).Ytest = tmp(:, 1:num_test_samples);
    Div_data(division_idx).Yanc  = tmp(:, num_test_samples+1:end);
    Div_data(division_idx).B = Div_data(division_idx).Yanc * pinv(X_anchor);
end

intermediate_dims = size(tmp, 1);

end