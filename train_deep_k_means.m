rng = RandStream.getGlobalStream;
rng.reset();

p_size = '32';              % patch size
l_type = 'k_means';         % k-means or omp1
t_type = 'whole';           % inpaint or whole

dir_in = 'test_data/';
input_name = 'patches.mat';

dir_train_data = '/test_data';
train_data_name = 'patches_64.mat';

dir_out = 'test_data/';
output_name = strcat(l_type,'_features_',p_size,'.mat');

addpath(dir_in);
addpath(dir_out);


%% Load Aline patches
patches = load(strcat(dir_in,input_name));
patches = patches.patches;


%% Initialize parameters
num_patches = size(patches,1);
dim_patches = size(patches,2);
patch_size = sqrt(dim_patches);         % patch size
patch_size = [patch_size, patch_size];


%% preprocess patches
% contrast normalization
fprintf('Contrast Normalization step...\n');
eps1 = 10;
patches = contrast_normalization(patches, eps1);

% ZCA whitening
fprintf('ZCA Whitening step...\n');
eps2 = 0.1;
[patches, ZCA, mean_patches] = zca_whitening(patches, eps2);


%% 1st layer learning
fprintf('1st layer learning...\n');

%% run feature learning
fprintf('Feature learning step...\n');

iter = 200;                 % number of K-means iterations
batch_size = 1000;          % batch size, useful for parallel computation??
K = 100;                    % number of centroids to learn

% feature learning
if (strcmp(l_type,'k_means')==1)
    tic
    D_1 = kmeans_learning(patches, K, iter, batch_size);
    t = toc;
    fprintf('Running time for K-means learning: %f\n',t);
elseif (strcmp(l_type,'omp1')==1)
    tic
    D_1 = omp1(patches, K, iter, batch_size);
    t = toc;
    fprintf('Running time for OMP-1 learning: %f\n',t);
end

% save results
save(strcat(dir_out,output_name),'D_1','ZCA','mean_patches','patches','-v7.3');
clear

% HERE:: Create a new dataset for feature learning!!!!!
% Every image should have the same size!!!!

% feature extraction
fprintf('Feature extraction step...\n');

% dimensions of the new larger image patches, should be the same size for
% every patch!!!!
dims = [64,64];
rf_size = [32,32];                  % receptive field size of the 1st layer
eps1 = 10;                          % epsilon for ZCA whitening, same as the 1st layer
pool_size = [4,4];                % pool window size for feature pooling     
activation_type = 'abs';            % type of activation function

% load data, CHANGE THIS LATER!!!!!
load(output_name,'D_1','mean_patches','ZCA');
load(train_data_name,'patches');
[n,d] = size(patches);

num_groups = 10;            % number of groups for filter clustering
k = 10;                     % number of neighbors for graph construction
type = 2;                   % mutual graph
sigma = 0;                  % binary graph

% cluster the learned filters of the 1st layer
fprintf('Filter clustering...\n');
groups = cluster_filters(D_1, num_groups, k, type, sigma);

% run feature extraction with max-pooling both on pixel coordinates and
% filters

% dimensions of one max-pooled feature
dim1 = round((dims(1)-rf_size(1)+1)/pool_size(1));
dim2 = round((dims(2)-rf_size(2)+1)/pool_size(2));

% sequential feature extraction and filter pooling for each input
% image/patch
pooled_features = zeros(n,dim1*dim2*num_groups);
%pooled_features = zeros(n,dim1*dim2*K);
fprintf('Main feature extraction...\n');
tic
for p = 1:n
    [features, ~, ~] = feature_extraction(patches(p,:), D_1, dims, rf_size, mean_patches, ZCA, eps1, pool_size, activation_type);
    pooled_features(p,:) = group_pooling(features, dim1, dim2, D_1, groups, num_groups);
end
t = toc

%% 2nd layer learning
fprintf('2nd layer learning...\n');

% Same thing here as in the 1st layer learning!!!!

% run feature learning
fprintf('Feature learning step...\n');

iter = 500;                 % number of K-means iterations
batch_size = 500;           % batch size, useful for parallel computation??
K = 100;                    % number of centroids to learn

% feature learning
if (strcmp(l_type,'k_means')==1)
    tic
    D_2 = kmeans_learning(pooled_features, K, iter, batch_size);
    t = toc;
    fprintf('Running time for K-means learning: %f\n',t);
elseif (strcmp(l_type,'omp1')==1)
    tic
    D_2 = omp1(pooled_features, K, iter, batch_size);
    t = toc;
    fprintf('Running time for OMP-1 learning: %f\n',t);
end

% feature extraction
fprintf('Feature extraction step...\n');

num_groups = 10;            % number of groups for filter clustering
k = 10;                     % number of neighbors for graph construction
type = 2;                   % mutual graph
sigma = 0;                  % binary graph

% cluster the learned filters of the 1st layer
fprintf('Filter clustering...\n');
groups = cluster_filters(D_2, num_groups, k, type, sigma);

% run feature extraction with max-pooling both on pixel coordinates and
% filters, matlab vectorization
%features_2_n = (D_2 * pooled_features')';

% the same as before, in a more sequential way, with group pooling
features_2 = zeros(n,K);
pooled_features_2 = zeros(n,num_groups);
for f = 1:n
    features_2(f,:) = (D_2 * pooled_features(f,:)')';
    pooled_features_2(f,:) = group_pooling(features_2(f,:), 1, 1, D_2, groups, num_groups);
end