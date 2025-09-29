function [laplacian_matrix, degree_matrix] = compute_laplacian_decomposition(data_matrix)

num_features = size(data_matrix,1);
num_samples = size(data_matrix,2);
weight_matrix = zeros(num_samples);
k_nearest_neighbors = min(7, num_samples-1);

squared_norms = sum(data_matrix.^2,1);
distance_matrix = repmat(squared_norms, num_samples, 1) + repmat(squared_norms', 1, num_samples) - 2*data_matrix'*data_matrix;
% weight_matrix = exp(-distance_matrix / (2*sigma^2));

% Auto-tuning for sigma
[sorted_distances, distance_indices] = sort(distance_matrix, 1);
knn_distances = max(sorted_distances(k_nearest_neighbors+1,:), 0);
local_scales = sqrt(knn_distances);
local_scales = local_scales' * local_scales;
valid_scales = (local_scales~=0);
weight_matrix(valid_scales) = exp(-distance_matrix(valid_scales) ./ (2*local_scales(valid_scales)));

weight_matrix = weight_matrix - eye(size(weight_matrix));

% k-nearest neighbor
for i = 1:num_samples
    w = weight_matrix(:,i);
    [~,id] = sort(w,'descend');
    w(id((k_nearest_neighbors+1):end)) = 0;
    weight_matrix(:,i) = w;
end
weight_matrix = (weight_matrix + weight_matrix')/2;

degree_matrix = diag(sum(weight_matrix));
laplacian_matrix = degree_matrix - weight_matrix;

end
