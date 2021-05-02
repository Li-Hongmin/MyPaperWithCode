clear
load 'Datasets/jaffe.mat'
%%
k = length(unique(gnd));
ensemble_size = 100;
kernels = preELSC(fea', ensemble_size);
label = ELSC(kernels, k);
%%
addpath('../utils/')
acc = metric_acc(gnd, label)
nmi = metric_nmi(gnd, label)