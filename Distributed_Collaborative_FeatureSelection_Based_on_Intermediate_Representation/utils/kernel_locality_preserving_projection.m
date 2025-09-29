function [Ytest,Ytrain] = kernel_locality_preserving_projection(Xtest,Xtrain,param)

[L,D] = compute_laplacian_decomposition(Xtrain);
[K,K2] = compute_kernel_matrix(Xtrain,Xtest,param);

[x,lambda] = eig(L,D);

lambda = diag(lambda);
% dd = sum(abs(lambda)<param.delta3);
% dd = size(L,1);
dd = min(param.neig,size(L,1))-1;
[~,ind] = sort(abs(lambda));

delta = lambda(ind(dd+1));
lambda = lambda(ind(1:dd));
x = x(:,ind(1:dd));

Ytrain = x * diag(1./(delta - 1./lambda).^param.delta2);

% Ytest = K2' * (K \ Ytrain);
Ytest = K2' * (pinv(K) * Ytrain);

Ytrain = Ytrain';
Ytest = Ytest';

end
