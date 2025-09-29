function [x,label] = kernel_locality_preserving_projection_distributed(Xtest,Xtrain,Xanc,ltrain,nd,param)
% Input
%    Xtest: test data
%    Xtrain: training data Xtrain = [Xtrain_1, Xtrain_2, ..., Xtrain_nd]
%    Xanc: anchor data
%    ltrain: label for training data ltrain = [ltrain_1, ltrain_2, ..., ltrain_nd]
%    nd: number of divisions
%    param: parameter

ntrain = size(Xtrain,2);
ntest = size(Xtest,2);
nanc = size(Xanc,2);

for i = 1:nd
    is = (i-1)*ntrain/nd+1; ie = i*ntrain/nd;
    
    Div_data(i).X = Xtrain(:,is:ie);
    Div_data(i).l = ltrain(is:ie);
end

% construct intermidiate replezantation by dimension reduction
% for each distributed data
for i = 1:nd
    [tmp, Div_data(i).Y] = kernel_locality_preserving_projection([Xtest,Xanc], Div_data(i).X, param);
    Div_data(i).Ytest = tmp(:,1:ntest);
    Div_data(i).Yanc  = tmp(:,ntest+1:end);
end
nmid = size(tmp,1);

% construct an optimal subspace
tmp = [];
for i = 1:nd
    tmp = [tmp; Div_data(i).Yanc];
end
[U,S,V] = svd(tmp,0);
% Z = S(1:nmid,1:nmid) * V(1:nmid,1:nanc);
Z = V(1:nmid,1:nanc);

% compute linear operators from each intermidiate replizantation to
% convined subspace
for i = 1:nd
    [U,S,V] = svd(Div_data(i).Yanc,0);
    dd = sum(diag(S)/S(1,1) >= 1e-10);
    Div_data(i).W = ((Z * V(:,1:dd)) / S(1:dd,1:dd)) * U(:,1:dd)';
end

% convert each intermidiate replizantation to convined subspace
Ztrain = [];
for i = 1:nd
    Div_data(i).Z = Div_data(i).W * Div_data(i).Y;
    Ztrain = [Ztrain, Div_data(i).Z];
end

% for the case that test data are not shared 
test_ind = randperm(nd,1);
Ztest = Div_data(test_ind).W * Div_data(test_ind).Ytest;

[x,label] = kernel_least_squares_classification(Ztest,Ztrain,ltrain,param);

end