function [scores, idcs] = hubness(data, k)
    % hubness score of the dataset

    n = size(data, 1);

    if n <= k
        scores = ones(n, 1);
        idcs =[];
        return
    end

    nsqRows = sum(data.^2, 2);
    nsqCols = sum(data.^2, 2);
    D = bsxfun(@minus, nsqRows, data * (2 * data'));
    D = bsxfun(@plus, nsqCols', D);
    idcs = zeros(n, k);
    D = D + eye(n) * 1e100;

    for i = 1:k
        [~, idcs(:, i)] = min(D, [], 2);
        temp = (idcs(:, i) - 1) * n + [1:n]';
        D(temp) = 1e100;
    end

    idcs = idcs(:, 1:k);
    % scores computing
    single_list = 1:n;
    scores = histc(idcs(:), single_list);
end
