function [x, label] = local_feature_selection(Xtest, Xtrain, ltrain, param)
%LOCAL_FEATURE_SELECTION Local feature selection using kernel locally linear projection
%
% Input:
%   Xtest  - Test data matrix (features × samples)
%   Xtrain - Training data matrix (features × samples)
%   ltrain - Training labels vector
%   param  - Parameter structure with algorithm settings
%
% Output:
%   x      - Selected feature results
%   label  - Classification results

ntrain = size(Xtrain,2);
mtrain = size(Xtrain,1);
ntest = size(Xtest,2);

%(1)
L = compute_laplacian_decomposition(Xtrain);   

% parameters setting
alpha =0.1;
beta = 0.1;
m = 10; % the number of low dimensions

%(2)
U = eye(mtrain); 
for i =1:10
    try
        %(3)
        A = Xtrain * Xtrain' + alpha *U;
        %(4)
        G = beta* L + eye(size(L))-Xtrain'*pinv(A)*Xtrain;
        OPTS.disp = 0;
        %(5)
        [Y, val] = eigs(L, m, 'SA', OPTS);
        % (6)
        W = pinv(A)*Xtrain*Y;
        %(7)
        for j = 1:mtrain
            U(j,j) = 0.5/norm(W(j,:),2);
            %problem
            if U(j,j) == Inf
                U(j,j) = 0;
            end
        end
    catch
        break
    end
end
%
for i = 1:mtrain
    rankvalue(i,1) = norm(W(i,:),2);
end
[temp,rank] = sort(rankvalue,'descend');

%% figure ranging of mi
figure
stem(rankvalue,'-o')
xlabel('number of features')
ylabel('ranging of norm of mi')
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
ax =gca;
ax.FontSize=16;
title('LFSweight');

for i = 1:mtrain
    [x{i},label{i}] = kernel_least_squares_classification(Xtest(rank(1:i,1),:),Xtrain(rank(1:i,1),:),ltrain,param);
end