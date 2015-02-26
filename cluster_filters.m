%%  C = cluster_filters(D, num_groups, k, type, sigma):
%%
%%  function that clusters the learned filters using Spectral Clustering.
%%
%%  Input:
%%      D: learned filters
%%      num_groups: number of groups for filter clustering
%%      k: number of nearest neighbors for Graph computation
%%      type: type of nearest neighbors graph
%%      sigma: parameter for Gaussian weighting of distances
%%
%%  Output:
%%      C: cluster assignments of each filter
%%
function C = cluster_filters(D, num_groups, k, type, sigma)

% build a similarity graph
W = SimGraph_NearestNeighbors(D', k, type, sigma);

% do spectral clustering
[C, L, U] = SpectralClustering(W, num_groups, 3);

end