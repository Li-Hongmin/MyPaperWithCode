%##############################################################################%
% Created Date: Monday December 30th 2019                                      %
% Author: Li Hongmin (li.hongmin.xa@alumni.tsukuba.ac.jp)                      %
%##############################################################################%

function [label_consensus, alpha0, consensus_eigenvector, basic_paritions_eigenvectors, ...
    obj_start, obj_val] = ELSC(kernels, n_cluster, n_iters, lambda, alpha, kmeans_rep)
    alpha0 =0;
    if ~exist('n_iters', 'var')
        n_iters = 10;
    end
    if ~exist('lambda', 'var')
        lambda = 1;
    end

    if ~exist('kmeans_rep', 'var')
        kmeans_rep = 10;
    end
    if ~exist('alpha', 'var')
        alpha = 0.01;
    end
    n_instaces = size(kernels{1}, 1);
    n_kernel = numel(kernels);
    L = cell(n_kernel, 1);
%     n_eigenvetors = round(sqrt(n_instaces));
    n_eigenvetors = n_cluster;
    basic_paritions_eigenvectors(n_instaces, n_eigenvetors, n_kernel, n_iters) = 0;
    obj_val = zeros(n_kernel, n_iters);

    opts.disp = 0;
    % opts.MaxIterations =550;

    %% obtain L matrix and its eigenvector basic_paritions_eigenvectors
    % from each kernel
    for i = 1:n_kernel
        % fprintf('computing kernel for X(%d)\n', i);
        K = kernels{i};
        D = diag(sum(K, 1));
        sD = sqrt((D));
        L{i} = sD \ K / sD;

        [eigVectors, eigValues] = eig(full(L{i}));
        eigValues = diag(eigValues);
        [eigValues, idx] = sort(eigValues, 'descend');
        nEigVec = eigVectors(:, idx(1:n_eigenvetors));
        sq_sum = sqrt(sum(nEigVec .* nEigVec, 2)) + 1e-20;
        nEigVec = nEigVec ./ repmat(sq_sum, 1, n_eigenvetors);
        basic_paritions_eigenvectors(:, :, i, 1) = real(nEigVec);
        obj_val(i, 1) = sum(eigValues(1:n_eigenvetors));
    end

    %% check NaNs
    [~, TF] = rmmissing(obj_val(:, 1));

    if sum(TF) > 0
        fprintf("kernel %d did not coverge, remove it!\n", find(TF));
        n_kernel = n_kernel - sum(TF);
        L(TF) = [];
        basic_paritions_eigenvectors(:, :, TF, :) = [];
        obj_val(TF, :) = [];
    end

    %% initialization of optimaztion
    n_kernel = numel(L);
    l_start = L{1};

    for j = 2:n_kernel
        l_start = l_start + L{j};
    end
%



    obj = sum(obj_val(:, 1));
    oldobj = obj - 1;

    obj_start(1:n_iters) = 0;
    %% start iterative optimaztion
    consensus_eigenvector(n_instaces, n_eigenvetors, n_iters) = 0;
    i = 1;

    while obj > oldobj && i < n_iters

        % fprintf('Running iteration %d\n', i - 1);
        % find consensusbasic_paritions_eigenvectorsU
        [consensus_eigenvector(:,:,i), Estar] = eigs(l_start, n_eigenvetors, 'LA', opts);
        obj_start(i) = sum(diag(Estar));



        % trun lambda
        lambda0 = std(real(obj_val(1:n_kernel, i)))^2 * alpha;

        i = i + 1;
        % optimaze basic_paritions_eigenvectors u
        for j = 1:n_kernel
            tmp = L{j} + lambda0 * l_start;

            [eigVectors, eigValues] = eig(full(tmp));
            eigValues = diag(eigValues);
            [eigValues, idx] = sort(eigValues, 'descend');
            nEigVec = eigVectors(:, idx(1:n_eigenvetors));
            sq_sum = sqrt(sum(nEigVec .* nEigVec, 2)) + 1e-20;
            nEigVec = nEigVec ./ repmat(sq_sum, 1, n_eigenvetors);
            basic_paritions_eigenvectors(:, :, j, i) = real(nEigVec);
            obj_val(j, i) = sum(eigValues(1:n_eigenvetors));
        end

        l_start(1:n_instaces, 1:n_instaces) = 0;

        for j = 1:n_kernel
            l_start = l_start + ...
                basic_paritions_eigenvectors(:, :, j, i) * basic_paritions_eigenvectors(:, :, j, i)';
        end

        oldobj = obj;
        obj = lambda *sum(obj_val(:, i)) +  sum(diag(Estar));

    end

    [consensus_eigenvector(:,:,i), Estar] = eigs(l_start, n_eigenvetors, 'LA', opts);

    consensus_eigenvector(:,:,i) = real(consensus_eigenvector(:,:,i));
    normvect = sqrt(diag(consensus_eigenvector(:,:,i) * consensus_eigenvector(:,:,i)'));
    normvect((normvect == 0.0)) = 1;
    consensus_eigenvector(:,:,i) = diag(normvect) \ consensus_eigenvector(:,:,i);

    label_consensus = kmeans(consensus_eigenvector(:,:,i) , n_cluster, 'Replicates',3);

    consensus_eigenvector(:,:,i:end) = [];
    basic_paritions_eigenvectors(:, :, :, i:end) = [];
    obj_start(i:end) = [];
    obj_val(:,i:end)= [];
end
