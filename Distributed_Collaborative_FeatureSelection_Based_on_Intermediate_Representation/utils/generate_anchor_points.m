function X_anchor_generated = generate_anchor_points(num_anchors, X_train)
%GENERATE_ANCHOR_POINTS Generate random anchor points based on training data distribution
%
% Input:
%   num_anchors - Number of anchor points to generate
%   X_train     - Training data matrix (features × samples)
%
% Output:
%   X_anchor_generated - Generated anchor points matrix (features × num_anchors)

X_mean = sum(X_train,2) / size(X_train,2);
X_std = sqrt(sum((X_train - X_mean).^2,2)/ size(X_train,2));

X_center = (max(X_train,[],2) + min(X_train,[],2)) / 2;
X_radius = (max(X_train,[],2) - min(X_train,[],2)) / 2;

% Generate anchors within the data range
% Alternative methods (commented):
% X_anchor_generated = X_mean * ones(1,num_anchors) + 4 * (X_std * ones(1,num_anchors)) .* (rand(size(X_train,1),num_anchors) - 0.5);
% X_anchor_generated = (rand(size(X_train,1),num_anchors) - 0.5 + max(X_train')' - min(X_train')') .* (0.5 * (max(X_train')' + min(X_train')'));
X_anchor_generated = (X_center * ones(1,num_anchors)) + 2 * (X_radius * ones(1,num_anchors)) .* (rand(size(X_train,1),num_anchors) - 0.5);

end
