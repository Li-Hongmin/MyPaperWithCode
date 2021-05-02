%% read data
load Datasets / twomoons - 1M.mat
%% run DnC-SC clustering
k = length(unique(y));
tic;
labels = DnC_SC(X, k);
toc
%% show results
acc = metric_acc(y, labels)
nmi = metric_nmi(y, labels)
