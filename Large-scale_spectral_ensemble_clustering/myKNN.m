function [RpFeaKnnDist, RpFeaKnnIdxFull] = myKNN(fea, RpFea, Knn, fea_idx, RpFeaKnnIdx)
    %myKNN:
    %   version 1.0 --April 2021
    %   Written by Hongmin Li (li.hongmin.xa@alumni.tsukuba.ac.jp)
    %===========
    %myKNN    Accelerated KNN distance computation
    MaxDim = 500;
    [n, d] = size(fea);

    RpFeaKnnDist = zeros(n, Knn);

    RpFeaKnnIdxFull = RpFeaKnnIdx(fea_idx, :);
    numberOfZeros = sum(fea(1:10, :) == 0, 'all');

    if (numberOfZeros < 2 * d) && (d < MaxDim)
        d = @(x, y) sum((x - y).^2, 2);

        for i = 1:Knn
            RpFeaKnnDist(:, i) = d(fea, RpFea(RpFeaKnnIdxFull(:, i), :));
        end

    else
        p = max(fea_idx);

        if d < 100
            distance = @pdist2;
        else
            distance = @(x, y) EuDist2(x, y, 0);
        end

        for i = 1:p
            tmp = fea_idx == i;
            RpFeaKnnDist(tmp, :) = distance(fea(tmp, :), RpFea(RpFeaKnnIdx(i, :), :));
        end

    end

end
