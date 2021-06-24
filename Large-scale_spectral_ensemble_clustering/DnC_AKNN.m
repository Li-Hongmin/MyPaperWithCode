function [knnDist, knnIdx] = DnC_AKNN(fea, LdFea, fea_idx, knn)
%% distance computation for different dimensions
n = size(fea,1);

%% The condidate neighborhood size.
Knn = 10 * knn;
% LdFeaW = distance(LdFea, LdFea);
% LdFeaW = LdFeaW + 1e100*eye(p);
% LdKnnIdx = zeros(p, Knn);
% 
% for i = 1:Knn
%     [~, LdKnnIdx(:, i)] = min(LdFeaW, [], 2);
% %     temp = sparse(1:p, LdKnnIdx(:, i), true);
%     temp = (LdKnnIdx(:, i) - 1) * p + (1:p)';
% 
%     LdFeaW(temp) = 1e100;
% end
% % clear LdFeaW temp

[~, LdKnnIdx] = pdist2(LdFea, LdFea, 'euclidean', 'Smallest', Knn);


%% partital pairwise distance matrix
% only compute possible nearest landmarks

[LdFeaKnnDist, LdFeaKnnIdxFull] = myKNN(fea, LdFea, Knn, fea_idx, LdKnnIdx');

% clear fea LdFea fea_idx LdFeaKnnIdx
[knnDist, knnTmpIdx] = mink(LdFeaKnnDist, knn, 2);
knnIdx = zeros(n, knn);
% [knnDist, knnIdx] = deal(zeros(n, knn));
for i = 1:knn
%     [knnDist(:, i), knnTmpIdx(:,i)] = min(LdFeaKnnDist, [], 2);
%     tmp = sparse(1:n, knnTmpIdx, true);
    tmp = (knnTmpIdx(:,i) - 1) * n + (1:n)';
    knnIdx(:, i) = LdFeaKnnIdxFull(tmp);
%     tmp = sparse(knnTmpIdx(:,i), 1:n, true);
%     knnIdx(:, i) = LdFeaKnnIdxFull(tmp');
end
