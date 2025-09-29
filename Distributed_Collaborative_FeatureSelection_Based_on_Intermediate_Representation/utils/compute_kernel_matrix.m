function [kernel_train, kernel_test] = compute_kernel_matrix(X_train, X_test, algorithm_params)

num_train_samples = size(X_train,2);
num_test_samples = size(X_test,2);

if algorithm_params.ksigma == 0
    k_nearest_neighbors = min(7, num_train_samples-1);

    squared_norms = sum(X_train.^2,1);
    distance_matrix = repmat(squared_norms, num_train_samples, 1) + repmat(squared_norms', 1, num_train_samples) - 2*X_train'*X_train;
    [sorted_distances, distance_indices] = sort(distance_matrix, 1);
    knn_distances = max(sorted_distances(k_nearest_neighbors+1,:), 0);
    local_scales = sqrt(knn_distances);
    algorithm_params.ksigma = sqrt(local_scales * local_scales' / num_train_samples);
end

if algorithm_params.kernel == 'G'

    kernel_sigma = algorithm_params.ksigma;

    train_squared_norms = sum(X_train.^2,1);
    train_distance_matrix = repmat(train_squared_norms, num_train_samples, 1) + repmat(train_squared_norms', 1, num_train_samples) - 2*X_train'*X_train;
    kernel_train = exp(-train_distance_matrix / (2*kernel_sigma^2));

    test_squared_norms = sum(X_test.^2,1);
    cross_distance_matrix = repmat(test_squared_norms, num_train_samples, 1) + repmat(train_squared_norms', 1, num_test_samples) - 2*X_train'*X_test;
    kernel_test = exp(-cross_distance_matrix / (2*kernel_sigma^2));
    
%     kernel_test = zeros(num_train_samples, size(X_test,2));
%     for test_idx = 1:size(X_test,2)
%         for train_idx = 1:num_train_samples
%             kernel_test(train_idx, test_idx) = exp(-norm((X_train(:,train_idx) - X_test(:,test_idx)))^2 / (2*kernel_sigma^2));
%         end
%     end

elseif algorithm_params.kernel == 'L'

    kernel_train = X_train' * X_train;
    kernel_test = X_train' * X_test;

end

end