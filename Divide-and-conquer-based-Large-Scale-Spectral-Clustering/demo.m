%% read data
load 'Datasets/TM_1M.mat'
%% run DnC-SC clustering
k = length(unique(gnd));
tic;
labels = DnC_SC(fea, k);
toc
%% show results
addpath('../utils/')
acc = metric_acc(gnd, labels)
nmi = metric_nmi(gnd, labels)
