function [marks, mark_index, scores] = localhubs(x, ...
        num_mark, num_part, partitionmode, k, varargin)
    % litehubs will returns the datapoints with largest local hubness scores
    % paremeters:
    %   global x is the input data
    %   num_mark    the number of landmark
    %   num_part    the number of partitions m
    %   partitionmode  'kmeans' or 'random'
    %   k                  the number of knn
    %   varargin    the paremeters of litekmeans
    %-----------------------------------------------

    % global x
    % initail dividing
    n = size(x, 1);
    r = 3;

    switch partitionmode
        case 'kmeans'
            index = litekmeans(x, num_part, varargin{:});
        case 'random'
            % index = randi(num_part, n, 1);
            np = set_sub_k(repmat((n / num_part), 1, num_part));
            index = zeros(n, 1);
            j = 0;

            for i = 1:num_part
                index(j + 1:j + np(i)) = i;
                j = j + np(i);
            end

    end

    % compute the p_i for every part, where p_i is the number of landmark we
    % select in part i
    indexp = index == 1:num_part;
    p = sum(indexp) / n * num_mark;
    % round values while preserve the sum
%     p = round_preserve_sum(p);
    p = set_sub_k(p);

    % scores computing & mark index from selection
    [scores, scoresnet] = deal(zeros(size(index)));
    idcs = zeros(size(index, 1), k);
    mark_index = [];

    for i = 1:num_part
        % local identfy
        local_idx = find(indexp(:, i));
        local_data = x(local_idx, :);
        % hubness scores
        [local_scores, localidcs]  = hubness(local_data, k);
%         [local_scores, localidcs] = hubness(local_data, k);
%         idcs(local_idx, :) = localidcs
%         for j = 1: r
%             local_scores = local_scores + sum(local_scores(localidcs), 2);
%         end
        scores(local_idx, :) = local_scores;

        % selection
        if p(i) > 0 && strcmp(partitionmode, 'kmeans')
            % [~, sortlist] = sort(local_scores, 'descend');
            % ind = sortlist(1:p(i));
            % mark_index(end + 1:end + p(i)) = local_idx(ind);

            mark_index(end + 1:end + p(i)) = datasample(local_idx, p(i), 'Replace', false, 'Weights', local_scores ./ sum(local_scores));
        end

    end

    % selection
    switch partitionmode
        case 'kmeans'
            marks = x(mark_index, :);
        case 'random'
            [marks, mark_index] = datasample(x, num_mark, ...
                'Replace', false, 'Weights', scores ./ sum(scores));
            % [~, sortlist] = sort(scores, 'descend');
            % mark_index = sortlist(1:num_mark);
            % marks = x(mark_index, :);
    end
