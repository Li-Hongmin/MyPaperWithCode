function [Z, division_data, convergence_metrics_reg, convergence_metrics_unreg] = collaborative_optimization(Z, division_data, X_anchor, laplacian_matrix, intermediate_dims, num_anchor_points)
%COLLABORATIVE_OPTIMIZATION Iterative optimization for collaborative feature selection
%
% Input:
%   Z                  - Initial combined subspace representation
%   division_data      - Structure array with intermediate representations
%   X_anchor          - Anchor data matrix
%   laplacian_matrix  - Laplacian matrix for graph regularization
%   intermediate_dims - Number of dimensions in intermediate representation
%   num_anchor_points - Number of anchor points
%
% Output:
%   Z                         - Optimized combined subspace representation
%   division_data             - Updated structure array with optimized parameters
%   convergence_metrics_reg   - Optimization convergence metrics with regularization
%   convergence_metrics_unreg - Optimization convergence metrics without regularization

num_divisions = length(division_data);
max_iter = 10;
convergence_tol = 1e-6;
alpha = 0.2;  % Regularization parameter
beta = 0.2;   % Graph regularization parameter

% Pre-allocate for performance
convergence_metrics_reg = zeros(max_iter, num_divisions);
convergence_metrics_unreg = zeros(max_iter, num_divisions);
Z_previous = Z;

for iter = 1:max_iter
    objective_diff_reg = zeros(1, num_divisions);
    objective_diff_unreg = zeros(1, num_divisions);

    for division_idx = 1:num_divisions
        objective_diff_reg(division_idx) = norm(Z - division_data(division_idx).M' * X_anchor, 'fro') + alpha * norm(division_data(division_idx).M', 2);
        objective_diff_unreg(division_idx) = norm(Z - division_data(division_idx).M' * X_anchor, 'fro');

        % Compute diagonal matrix U
        num_features = size(X_anchor, 1);  % Number of features (7129)
        diagonal_weights = zeros(1, num_features);
        for feature_idx = 1:num_features
            diagonal_weights(feature_idx) = 0.5 / norm(division_data(division_idx).M(feature_idx, :), 2);
            if diagonal_weights(feature_idx) == Inf
                diagonal_weights(feature_idx) = 0;
            end
        end

        try
            division_data(division_idx).U = diag(diagonal_weights);

            % Update M using optimized stable formulation
            A = X_anchor*X_anchor' + alpha*division_data(division_idx).U + beta*X_anchor*laplacian_matrix*X_anchor';

            % Add progressive regularization for numerical stability
            base_reg = 1e-4;
            A = A + base_reg * eye(size(A));

            % Progressive regularization based on condition number
            current_rcond = rcond(A);
            if current_rcond < 1e-12
                A = A + 1e-2 * eye(size(A));  % Strong regularization
            elseif current_rcond < 1e-8
                A = A + 1e-3 * eye(size(A));  % Medium regularization
            end

            division_data(division_idx).M = A \ (X_anchor * Z');

            % Update B using backslash for better performance
            % Check if W is well-conditioned
            if rcond(division_data(division_idx).W) < 1e-12
                division_data(division_idx).W = division_data(division_idx).W + 1e-6 * eye(size(division_data(division_idx).W));
            end

            % Fix dimension: W \ M' to handle (9x9) \ (9x7129) = (9x7129), then transpose
            division_data(division_idx).B = (division_data(division_idx).W \ division_data(division_idx).M')';

            % Fix matrix multiplication: B' * X_anchor for (9x7129) * (7129x20) = (9x20)
            [svd_U, svd_S, svd_V] = svd(division_data(division_idx).B' * X_anchor, 0);
            Z = svd_V(1:intermediate_dims, 1:num_anchor_points);

            % Robust singular value selection
            num_significant_svd = sum(diag(svd_S)/svd_S(1,1) >= 1e-10);
            % Ensure num_significant_svd is valid
            if num_significant_svd < 1
                num_significant_svd = 1;
            end
            if num_significant_svd > size(svd_S, 1)
                num_significant_svd = size(svd_S, 1);
            end

            division_data(division_idx).W = ((Z * svd_V(:, 1:num_significant_svd)) / svd_S(1:num_significant_svd, 1:num_significant_svd)) * svd_U(:, 1:num_significant_svd)';
            division_data(division_idx).M = (division_data(division_idx).W * division_data(division_idx).B')';
        catch optimization_error
            fprintf('Optimization failed at iteration %d, division %d: %s\n', iter, division_idx, optimization_error.message);
            % Continue with next division instead of breaking
            continue;
        end
    end

    convergence_metrics_reg(iter, :) = objective_diff_reg;
    convergence_metrics_unreg(iter, :) = objective_diff_unreg;

    % Check for convergence
    if iter > 1
        Z_change = norm(Z - Z_previous, 'fro') / norm(Z_previous, 'fro');
        if Z_change < convergence_tol
            fprintf('Converged after %d iterations (change: %.2e)\n', iter, Z_change);
            % Trim unused rows
            convergence_metrics_reg = convergence_metrics_reg(1:iter, :);
            convergence_metrics_unreg = convergence_metrics_unreg(1:iter, :);
            break;
        end
    end
    Z_previous = Z;
end

end