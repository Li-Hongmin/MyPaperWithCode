function [subspace_matrix, Div_data] = construct_optimal_subspace(Div_data, intermediate_dims, num_anchor_points)
%CONSTRUCT_OPTIMAL_SUBSPACE Build optimal subspace and linear operators
%
% Input:
%   Div_data - Structure array with intermediate representations
%   intermediate_dims     - Number of dimensions in intermediate representation
%   num_anchor_points     - Number of anchor points
%
% Output:
%   subspace_matrix        - Combined subspace representation
%   Div_data - Updated structure array with linear operators

num_divisions = length(Div_data);

% Construct optimal subspace
tmp = [];
for division_idx = 1:num_divisions
    tmp = [tmp; Div_data(division_idx).Yanc];
end
[~, S, V] = svd(tmp, 0);
subspace_matrix = V(1:intermediate_dims, 1:num_anchor_points);

% Compute linear operators from each intermediate representation to combined subspace
for division_idx = 1:num_divisions
    [U, S, V] = svd(Div_data(division_idx).Yanc, 0);
    significant_dims = sum(diag(S)/S(1,1) >= 1e-10);
    Div_data(division_idx).W = ((subspace_matrix * V(:, 1:significant_dims)) / S(1:significant_dims, 1:significant_dims)) * U(:, 1:significant_dims)';
end

% Convert each intermediate representation to combined subspace
for division_idx = 1:num_divisions
    Div_data(division_idx).Z = Div_data(division_idx).W * Div_data(division_idx).Y;
    Div_data(division_idx).M = (Div_data(division_idx).W * Div_data(division_idx).B)';
end

end