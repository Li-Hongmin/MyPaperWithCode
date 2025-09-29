function [n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size)

if data_set == 'mnist'

n0 = 60000;
load 'Orig.mat';
switch data_size
    case 'small'
        n_train = 10000;
        n_test = 10000;
    otherwise
        n_train = 60000;
        n_test = 10000;
end
n_labels = 10;
wx = 28; wy = 28;


X_train = fea(1:n_train,:)';
Y_train = full(sparse(gnd(1:n_train)'+1,1:n_train,ones(1,n_train)));
X_test = fea(n0+(1:n_test),:)';
Y_test = full(sparse(gnd(n0+(1:n_test))'+1,1:n_test,ones(1,n_test)));

elseif data_set == 'cifar'

n_labels = 10;
wx = 32; wy = 32; rgb = 3;

load './DATA_SET/cifar-10-batches-mat/batches.meta.mat';
% label_names

load './DATA_SET/cifar-10-batches-mat/data_batch_1';
n_train = size(data,1);
%n_train = 2000;
labels = double(labels(1:n_train));
X_train = double(data(1:n_train,:)');
Y_train = full(sparse(labels'+1,1:n_train,ones(1,n_train),n_labels,n_train));

load './DATA_SET/cifar-10-batches-mat/data_batch_2';
labels = double(labels(1:n_train));
X_train = [X_train double(data(1:n_train,:)')];
Y_train = [Y_train full(sparse(labels'+1,1:n_train,ones(1,n_train),n_labels,n_train))];

load './DATA_SET/cifar-10-batches-mat/data_batch_3';
labels = double(labels(1:n_train));
X_train = [X_train double(data(1:n_train,:)')];
Y_train = [Y_train full(sparse(labels'+1,1:n_train,ones(1,n_train),n_labels,n_train))];

load './DATA_SET/cifar-10-batches-mat/data_batch_4';
labels = double(labels(1:n_train));
X_train = [X_train double(data(1:n_train,:)')];
Y_train = [Y_train full(sparse(labels'+1,1:n_train,ones(1,n_train),n_labels,n_train))];

load './DATA_SET/cifar-10-batches-mat/data_batch_5';
labels = double(labels(1:n_train));
X_train = [X_train double(data(1:n_train,:)')];
Y_train = [Y_train full(sparse(labels'+1,1:n_train,ones(1,n_train),n_labels,n_train))];

load './DATA_SET/cifar-10-batches-mat/test_batch';
n_test = size(data,1);
%n_test = 500;
labels = double(labels(1:n_test));
X_test = double(data(1:n_test,:)');
Y_test = full(sparse(labels'+1,1:n_test,ones(1,n_test),n_labels,n_test));

end

end
