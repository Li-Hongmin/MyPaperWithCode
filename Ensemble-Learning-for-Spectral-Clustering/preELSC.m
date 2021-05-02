%##############################################################################%
% Created Date: Tuesday December 31st 2019                                     %
% Author: Li Hongmin (li.hongmin.xa@alumni.tsukuba.ac.jp)                      %
%##############################################################################%
function kernels = preELSC(X, n)
    kernels = cell(0);

    kernels = cell(0);

    while numel(kernels) < n

        coef = 0.5 + rand(1);

        Dis = squareform(pdist(X'));

        for k = 5:9
            kernels = [kernels, knn(Dis, k, coef)];
        end

        Dis = 1 - corr(X, 'type', 'p');

        for k = 5:9
            kernels = [kernels, knn(Dis, k, coef)];
        end

        Dis = 1 - corr(X, 'type', 's');

        for k = 5:9
            kernels = [kernels, knn(Dis, k, coef)];
        end

        isnan = zeros(1, n);

        for i = 1:numel(kernels)
            TF = ismissing(kernels{i}) + isinf(kernels{i});
            isnan(i) = sum(TF(:));
        end

        if sum(isnan) > 0
            fprintf("kernel %d did not coverge, remove it!\n", find(isnan));
            kernels(logical(isnan)) = [];
        end

    end
end

    function Z = bestsaprse(W, coefs)
        % epsilon nearest neighbor 

        n = size(W, 1);
        n_unit = 100;


        maxedges = max(W(:));

        BinEdges = 0.03 * maxedges + 0.07 * maxedges * rand(10, 1);
        i = 3;
        Z = cell(0);
        sigma = mean(W(:)) * coefs;
        GKenerl = exp(-W / (2 * sigma^2));
        nnzs = 0;

        while i < 10
            s = W < BinEdges(i);
            nnzs = nnz(s);

            if nnzs <= n
                i = i + 1;
                continue
            end

            s = logical(s + s');
            i = i + 1;
            zi = GKenerl .* s;
            zi = bsxfun(@rdivide, zi, sum(zi, 2));
            Z = [Z, {zi}];
        end

    end

    function Z = knn(D, k, coef)
        % k-nearest neighbor 

        if ~exist('coef', 'var')
            coef = 1;
        end

        [nSmp, p] = size(D);

        for i = 1:k
            [dump(:, i), idx(:, i)] = min(D, [], 2);
            temp = (idx(:, i) - 1) * nSmp + [1:nSmp]';
            D(temp) = 1e100;
        end

        sigma = mean(dump(:)) * coef;
        dump = exp(-dump / (2 * sigma^2));
        sumD = sum(dump, 2) +1e-10;
        Gsdx = bsxfun(@rdivide, dump, sumD);
        Gidx = repmat([1:nSmp]', 1, k);
        Gjdx = idx;
        Z = sparse(Gidx(:), Gjdx(:), Gsdx(:), nSmp, p);
        Z = full(Z + Z') / 2;
    end
