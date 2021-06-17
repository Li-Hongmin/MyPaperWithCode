function B = gaussianKernel(knnDist, knnIdx, knn, p)

n = size(knnDist,1);

knnMeanDiff = mean(knnDist(:,1:knn), 'all');
Gsdx = exp(-(knnDist(:,1:knn).^2)/ (2 * knnMeanDiff^2)); 
% clear knnDist knnMeanDiff
Gsdx(Gsdx == 0) = eps;
Gidx = repmat((1:n)', 1, knn);
knnIdx = knnIdx(:,1:knn);
B = sparse(Gidx(:), knnIdx(:), Gsdx(:), n, p);
end