%%  D = omp1(X, K, iter):
%%
%%  function that runs OMP1. Very similar to K-means learning.
%%
%%  Input:
%%      X: training data, in our case whitened patches
%%      K: number of features to be learned
%%      iter: number of iterations
%%      batch_size: batch size for sequential learning
%%
%%  Output:
%%      D: the filters learned from OMP-1.
%%
function D = omp1(X, K, iter, batch_size)

% size of the dataset
[n,d] = size(X);

% initialize dictionary and normalize the rows to unit norm
D = 0.1 * randn(K,d);
D = bsxfun(@rdivide,D,sqrt(sum(D.^2,2)+1e-20));

% start OMP1 by alternate optimization of dictionary and sparse codes
for i=1:iter
    fprintf(1,'Running GSVQ: iteration = %d...\n', i);

    % do assignment + accumulation
    [summation, counts] = gsvq_step(X, D, batch_size);

    % re-initialize empty clusters
    I = find(sum(summation.^2,2) < 0.001);
    summation(I,:) = randn(length(I),d);

    % normalize the rows of the dictionary to unit norm
    D = bsxfun(@rdivide, summation, sqrt(sum(summation.^2,2)+1e-20));
end

end

%% Optimize over Dictionary and sparse codes  
function [summation, counts] = gsvq_step(X, D, batch_size)

% size of the dictionary
[K,d] = size(D);

summation = zeros([K,d]);
counts = zeros(K,1);

% go through batches
for i=1:batch_size:size(X,1)
    % compute the last index of the current batch
    lastInd=min(i+batch_size-1, size(X,1));
    
    % number of data in the batch (???)
    m = lastInd - i + 1;

    % optimize over the dictionary (similar to cluster assignments in
    % K-means)
    dots = D * X(i:lastInd,:)';
    [~,labels] = max(abs(dots));    % get labels

    % optimal assignments as indicator matrix
    E = sparse(labels,1:m,1,K,m,m);
    counts = counts + sum(E,2);     % sum up counts

    % update the dictionary
    dots = dots .* E;       % non-zero values only only on the argmax positions, elsewhere zero
    summation = summation + dots * X(i:lastInd,:); 
end

end
