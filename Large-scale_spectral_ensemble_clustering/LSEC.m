function labels = E2SC(fea, k, m, knnRange, nkeachEmbeding,...
    upperBound, lowerBound)
% generate m sets of landmark points


n = size(fea,1);
p = 1000;
q = numel(knnRange);
alpha = 50;
KmIters = 5;
KmRps = 1;
clusterN = randi([lowerBound, upperBound], 1, m*q);
BaseCl = zeros(n, m * q * nkeachEmbeding);
% nkeachEmbeding = numel(clusterRange)/m/numel(knnRange);

ii = 1;
q = numel(knnRange);

msg = ['line19: Starting E2SC with parameters k: ', int2str(k),...
    '; m: ', int2str(m), '; knnrange: ', int2str(knnRange),...
    ';nkeachEmbeding: ', int2str(nkeachEmbeding), '; upperBound: ', int2str(upperBound), ...
    '; lowerBound: ', int2str(lowerBound)];
%logIt(msg);

for i = 1:m
    msg = ['line26 First loop ', int2str(i), '/', int2str(m)];
    %logIt(msg);
    [fea_idx, LdFea] = DnC_landmark(fea, p, 10*p, alpha);
    
    msg = 'line29 First loop - DnC_AKNN';
    %logIt(msg);
    [knnDist, knnIdx] = DnC_AKNN(fea, LdFea, ...
        fea_idx, knnRange(q));
    %clear fea_idx LdFea
    
    % Generate multiple similarity matrix for the same knn graph
%     msg = 'line37 First loop - KNN';
    %logIt(msg);
%     [knnDist, knnIdx] = KNN(LdFeaKnnDist, LdFeaKnnIdxFull, knnRange);
%     %clear LdFeaKnnDist LdFeaKnnIdxFull
    
    
    for j = q:-1:1
        msg = ['line44 Second loop - gaussianKernel: ', int2str(clusterN)];
        %logIt(msg);
        B = gaussianKernel(knnDist, knnIdx, knnRange(j), p);
        % Generate a full embedding for the same simlarity
        msg = ['48 Second loop - myBipartiteGraphParitioin: ', int2str(clusterN)];
        %logIt(msg);
        BaseCl(:,ii) = myBipartiteGraphParitioin(B, clusterN(ii), KmIters, KmRps);
        fprintf(' %d',ii)
        ii = ii + 1;

    end
end
fprintf(' \n')

labels = USENC_ConsensusFunction(BaseCl,k);

function labels = USENC_ConsensusFunction(baseCls,k,maxTcutKmIters,...
    cntTcutKmReps)
% Huang Dong. Mar. 20, 2019.
% Combine the M base clusterings in baseCls to obtain the final clustering
% result (with k clusters).

if nargin < 4
    cntTcutKmReps = 3;
end
if nargin < 3
    maxTcutKmIters = 100; % maxTcutKmIters and cntTcutKmReps are used to limit the iterations of the k-means discretization in Tcut.
end

[N,M] = size(baseCls);

maxCls = max(baseCls);
for i = 1:numel(maxCls)-1
    maxCls(i+1) = maxCls(i+1)+maxCls(i);
end


cntCls = maxCls(end);
baseCls(:,2:end) = baseCls(:,2:end) + repmat(maxCls(1:end-1),N,1);
%clear maxCls
% Build the bipartite graph.
B=sparse(repmat((1:N)',1,M),baseCls(:),1,N,cntCls); %clear baseCls
colB = sum(B);
B(:,colB==0) = [];

% Cut the bipartite graph.
labels = myBipartiteGraphParitioin(B,k,maxTcutKmIters,cntTcutKmReps);
