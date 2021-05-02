function labels = myBipartiteGraphParitioin(B, Nseg, maxKmIters, cntReps)
    % B - |X|-by-|Y|, cross-affinity-matrix

    if nargin < 4
        cntReps = 3;
    end

    if nargin < 3
        maxKmIters = 100;
    end

    [Nx, Ny] = size(B);

    if Ny < Nseg
        error('Need more columns!');
    end

    dx = sum(B, 2);
    dx(dx == 0) = 1e-10; % Just to make 1./dx feasible.
    Dx = sparse(1:Nx, 1:Nx, 1 ./ dx); clear dx
    Wy = B' * Dx * B;

    %%% compute Ncut eigenvectors
    % normalized affinity matrix
    d = sum(Wy, 2);
    D = sparse(1:Ny, 1:Ny, 1 ./ sqrt(d)); clear d
    nWy = D * Wy * D; clear Wy
    nWy = (nWy + nWy') / 2;

    % computer eigenvectors

    [evec, eval] = eig(full(nWy)); clear nWy
    [~, idx] = sort(diag(eval), 'descend');
    Ncut_evec = D * evec(:, idx(1:Nseg)); clear D

    %%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
    evec = B * Ncut_evec; clear B Dx Ncut_evec

    %     [evec, eval] = eig(full(nWy)); clear nWy
    %     [eval, idx] = sort(diag(eval), 'descend');
    %     Ncut_evec = D * evec(:, idx(2:Nseg+1)); clear D
    %     lambda = eval(2:Nseg+1);
    %     r = -2*sqrt(1+lambda) -1;
    %
    %
    %     %%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
    %     evec = Dx * B * Ncut_evec * diag(1./(1-r)); clear B Dx Ncut_evec

    % normalize each row to unit norm
    evec = bsxfun(@rdivide, evec, sqrt(sum(evec .* evec, 2)) + 1e-10);

    % k-means
    %     labels = kmeans(evec,Nseg);
    if maxKmIters > 5
        labels = kmeans(evec, Nseg, 'MaxIter', maxKmIters, 'Replicates', cntReps);
    else
        labels = litekmeans(evec, Nseg, 'MaxIter', maxKmIters, 'Replicates', cntReps);
    end

end
