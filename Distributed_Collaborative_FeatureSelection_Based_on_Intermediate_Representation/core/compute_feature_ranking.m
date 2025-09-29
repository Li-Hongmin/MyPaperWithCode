function [feature_ranking, ranking_values] = compute_feature_ranking(Div_data, display_plot)
%COMPUTE_FEATURE_RANKING Compute feature ranking based on collaborative weights
%
% Input:
%   Div_data     - Structure array with optimized parameters
%   display_plot - Boolean flag to display ranking plot (default: false)
%
% Output:
%   feature_ranking      - Feature ranking indices (best to worst)
%   ranking_values - Feature importance scores

if nargin < 2
    display_plot = false;
end

num_divisions = length(Div_data);
test_division_idx = randperm(num_divisions, 1);  % Randomly select one division for ranking

num_features = size(Div_data(test_division_idx).M, 1);
ranking_values = zeros(num_features, 1);

% Compute feature importance scores
for feature_idx = 1:num_features
    ranking_values(feature_idx) = norm(Div_data(test_division_idx).M(feature_idx, :), 2);
end

% Sort features by importance (descending)
[~, feature_ranking] = sort(ranking_values, 'descend');

% Display ranking plot if requested
if display_plot
    figure;
    stem(ranking_values, '-o');
    xlabel('Feature Index');
    ylabel('Feature Importance (L2 norm)');
    set(findall(gca, 'Type', 'Line'), 'LineWidth', 2);
    ax = gca;
    ax.FontSize = 16;
    title('CFS Feature Importance Ranking');
    grid on;
end

end