function [projected_features, predicted_labels] = kernel_least_squares_classification(X_test, X_train, labels_train, algorithm_params)

num_train_samples = size(X_train,2);
if algorithm_params.k == 2
    num_output_vars = 1;
    label_matrix = -(labels_train-1.5)*2;
else
    num_output_vars = algorithm_params.k;
    label_matrix = zeros(algorithm_params.k, num_train_samples);
    for class_idx = 1:algorithm_params.k
        label_matrix(class_idx, labels_train==class_idx) = 1;
        label_matrix(class_idx, labels_train~=class_idx) = 0;
    end
end

kernel_params = algorithm_params;
kernel_params.kernel = "G";
[kernel_train, kernel_test] = compute_kernel_matrix(X_train, X_test, kernel_params);

regularization_param = sqrt(algorithm_params.delta1);
% classification_weights = (kernel_train*kernel_train + regularization_param * eye(num_train_samples)) \ (kernel_train*label_matrix');
classification_weights = (kernel_train + regularization_param * eye(num_train_samples)) \ (label_matrix');

projected_features = kernel_test' * classification_weights;
if num_output_vars == 1
    predicted_labels = (projected_features < 0) + 1;
else
    [~, predicted_labels] = max(projected_features');
end

% x2 = K' * alpha;
% label = myknn(25,x',x2',ltrain);

end
