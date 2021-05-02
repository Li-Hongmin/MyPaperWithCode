function labels = DnC_SC(fea, k, p, alpha, knn, maxTcutKmIters,cntTcutKmReps)
    if nargin < 7
        cntTcutKmReps = 3; 
    end
    if nargin < 6
        maxTcutKmIters = 100; % maxTcutKmIters and cntTcutKmReps are used to limit the iterations of the k-means discretization in Tcut.
    end
    if nargin < 5
        knn = 5; % The number of nearest neighbors.
    end
    if nargin < 4
        % The number of up-bounder of DnC-keams
        if (size(fea,1) > 1000000)
            alpha = 50;
        else
            alpha = 200;
        end
    end
    if nargin < 3
        p = 1000; % number of representatives
    end

    %% distance computation for different dimensions
    [n,m] = size(fea);
    if m <100
        distance = @pdist2;
    else
        distance = @(x,y) EuDist2(x,y,0);
    end
        

    %% divide
    [fea_idx, RpFea] = DnC_landmark(fea, p, 10*p, alpha);

    %% The condidate neighborhood size.
    Knn = 10 * knn;
    RpFeaW = distance(RpFea, RpFea);
    RpFeaW = RpFeaW + 1e100*eye(p);
    RpKnnIdx = zeros(p, Knn);
    
    for i = 1:Knn
        [~, RpKnnIdx(:, i)] = min(RpFeaW, [], 2);
        temp = (RpKnnIdx(:, i) - 1) * p + (1:p)';
        RpFeaW(temp) = 1e100;
    end
    clear RpFeaW temp

    %% partital pairwise distance matrix
    % only compute possible nearest representative
    
    [RpFeaKnnDist, RpFeaKnnIdxFull] = myKNN(fea, RpFea, Knn, fea_idx, RpKnnIdx);
    
    clear fea RpFea fea_idx RpFeaKnnIdx

    [knnDist, knnTmpIdx, knnIdx] = deal(zeros(n, knn));

    for i = 1:knn
        [knnDist(:, i), knnTmpIdx(:, i)] = min(RpFeaKnnDist, [], 2);
        temp = (knnTmpIdx(:, i) - 1) * n + (1:n)';
        RpFeaKnnDist(temp) = 1e100;
        knnIdx(:, i) = RpFeaKnnIdxFull(temp);
    end

    clear knnTmpIdx temp nearestRepInRpFeaIdx RpFeaKnnIdxFull RpFeaKnnDist

    %% Gaussian kernal
    
    knnMeanDiff = mean(knnDist, 'all'); % use the mean distance as the kernel parameter $\sigma$
    Gsdx = exp(-(knnDist.^2)/ (2 * knnMeanDiff^2)); clear knnDist knnMeanDiff
    Gsdx(Gsdx == 0) = eps;
    Gidx = repmat((1:n)', 1, knn);
    B = sparse(Gidx(:), knnIdx(:), Gsdx(:), n, p); clear Gsdx Gidx knnIdx
    % If a representative is not connected to any objects, then it will be removed.
    B(:, sum(B) == 0) = [];
    

    %% T cut
    labels = myBipartiteGraphParitioin(B, k, maxTcutKmIters,cntTcutKmReps);
    
end
