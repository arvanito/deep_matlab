%%  features = feature_extraction(X, D, dims, rf_size, mean_X, ZCA, eps1, pool_size, activation_type):
%%
%%  function that implements feature extraction with learned 
%%  features. 
%%
%%  Input:
%%      X: data matrix, rows correspond to points, columns to features
%%      D: learned centroids from the feature learning procedure
%%      dims: dimensions of (full) image data
%%      rf_size: receptive field size
%%      mean_X: mean points in training data matrix
%%      ZCA: ZCA matrix computed in the training data points
%%      eps1: epsilon parameter for ZCA whitening
%%      pool_size: size of the pooling window
%%      type: activation type
%%
%%  Output:
%%      features: final extracted convolutional features
%%      dim1: first dimension of the final pooled features
%%      dim2: second dimension of the final pooled features
%%
%%  TODO:: implement more activation functions here!!
function [features, dim1, dim2] = feature_extraction(X, D, dims, rf_size, mean_X, ZCA, eps1, pool_size, activation_type)

% size of the dataset
[n,d] = size(X);

% number of features learned
num_features = size(D,1);

% allocate memory for the final pooled features
% HERE: maybe this is floor!!
dim1 = round((dims(1)-rf_size(1)+1)/pool_size(1));
dim2 = round((dims(2)-rf_size(2)+1)/pool_size(2));

fprintf('Feature dimensionality: %d\n',num_features*dim1*dim2);
features = zeros(n,num_features*dim1*dim2);

% % count number of overlapping patches and their dimension
% num_patches = (dims(1) - rf_size(1) + 1) * (dims(2) - rf_size(2) + 1);
% dim_patches = length(mean_X);
    
% extract features and apply pooling for each data point
for i = 1:n
    %if (mod(i,1000) == 0) 
        fprintf('Extracting features: %d / %d\n', i, n); 
    %end
    % extract overlapping sub-patches from the current image
    % TODO:: Here do convolution instead!!!!
    patches = im2col(reshape(X(i,:),dims),rf_size)';
    
    % extract overlapping sub-patches into rows of 'patches'
    %patches = [ im2col(reshape(X(i,1:1024),dims(1:2)), [rf_size rf_size]) ;
    %            im2col(reshape(X(i,1025:2048),dims(1:2)), [rf_size rf_size]) ;
    %            im2col(reshape(X(i,2049:end),dims(1:2)), [rf_size rf_size]) ]';
    
    % alternative way to compute the above lines, more general, for color images 
    % CHANGE IT FOR GRAYSCALE ONES!!!!!
    % reshape it to look like an image
%     X_cur = reshape(X(i,:),dims);
%     patches = zeros(num_patches,dim_patches);
%     k = 1;
%     rf_size2 = rf_size(1) * rf_size(2);
%     
%     % for each channel
%     for c = 1:3 
%         % extract overlapping patches from current image channel
%         patches(:,k:k+rf_size2-1) = im2col(X_cur(:,:,c),rf_size)';
%         k = k + rf_size2;
%     end
    
    % contrast normalization
    patches = contrast_normalization(patches,eps1);
    
    % ZCA whitening
    patches = bsxfun(@minus,patches,mean_X);
    patches = patches * ZCA;
    
    % compute "triangle" activation function,
    % TODO:: put more activation functions here
    patches = compute_activation(patches, D, activation_type);
    
    % reshape to reflect convolutional feature extraction
    patches = reshape(patches,dims(1)-rf_size(1)+1,dims(2)-rf_size(2)+1,num_features);
    
    % apply pooling
    patches = pool(patches, pool_size);
    
    % final features by concatenation to a single vector 
    features(i,:) = patches(:)'; 
end

end