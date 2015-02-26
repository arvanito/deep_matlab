%%  D = kmeans_learning(X, K, iter, batch_size)
%%
%%  function that the K-means algorithm for feature learning.
%%
%%  Input:
%%      X: training data, in our case whitened patches
%%      K: number of features to be learned
%%      iter: number of iterations
%%      batch_size: batch size for sequential learning
%%
%%  Output:
%%      D: centroids learned from K-means. These correspond to 
%%         the learned features
%%
%% TODO:: Change convergence criterion!!!!
function D = kmeans_learning(X, K, iter, batch_size)

% training data size
[n,d] = size(X);

% initialize centroids
D = 0.1 * randn(K,d);

% precompute norm of training data
%norm_x = 0.5 * sum(X.^2);

% start K-means by alternate optimization of centroid 
% and cluster assignments
for i = 1:iter
    fprintf('K-means iteration %d / %d\n', i, iter);
        
    % compute norm of the centroids
    norm_D = 0.5 * sum(D.^2,2);
    
    % HERE!!
    summation = zeros(K,d);
    counts = zeros(K,1);
    
    % go through batches
    for b = 1:batch_size:n
        % compute the last index of the current batch
        last_index = min(b+batch_size-1,n);
        
        % number of data in the batch (???)
        m = last_index - b + 1;
        
        % optimize the cluster assignments, minimize square distance,
        % maximize similarity
        [~,labels] = max(bsxfun(@minus,D*X(b:last_index,:)',norm_D));
        
        % optimal assignments as indicator matrix
        S = sparse(1:m,labels,1,m,K,m); 
        
        % feature updates, sum up all the data that are closer to each
        % cluster
        summation = summation + S' * X(b:last_index,:);
        counts = counts + sum(S,1)';
    end
    
    % final features after i-th iteration, take their mean
    D = bsxfun(@rdivide,summation,counts);
    
    % remove empty clusters
    bad_index = (counts == 0);
    D(bad_index,:) = 0;
end

end