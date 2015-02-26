%%  pooled_features = cnn_pool(features, pool_size):
%%
%%  function that pool convolutional features.
%%
%%  Input: 
%%      features: features extracted after the convolution step
%%      pool_size: 2-d size used for the pooling
%%
%%  Output:
%%      pooled_features: features after pooling
%%
function pooled_features = pool(features, pool_size)

% convolved features size
[n,m,num_features] = size(features);

% allocate memory for the pooled features
pool_dim = [floor(n/pool_size(1)),floor(m/pool_size(2))];
pooled_features = zeros(pool_dim(1),pool_dim(2),num_features);

%% pool features together
% for each feature
for f = 1:num_features
    % loop over the new pooled dimensions
    k = 1;
    l = 1;
    for p = 1:pool_dim(1)
        for q = 1:pool_dim(2)
            % extract the patch indices
            kk = min(k+pool_size(1)-1,n);
            ll = min(l+pool_size(2)-1,m);
            patch_inds_k = k:kk;
            patch_inds_l = l:ll;
            
            % extract the patch to be pooled
            image_patch = squeeze(features(patch_inds_k,patch_inds_l,f));

            % pool the patch
            pooled_features(p,q,f) = max(image_patch(:));

            % update counter for the next patch on the right
            l = l + pool_size(2);
        end

        % update counters for the next patch below
        k = k + pool_size(1);
        l = 1;
    end
end

end