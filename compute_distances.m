%%
%%  function that computes L2 distances between points
%%  in two different datasets. The matrices X,Y contain in their 
%%  rows the data points.
%%
%%  Input:
%%      X: first dataset
%%      Y: second dataset
%%
%%  Output:
%%      dist: pair-wise distances between points in X and Y
%%
function dist = compute_distances(X, Y)

% L2 norms of first dataset
norm_x = sum(X.^2,2);

% L2 norms of second dataset
norm_y = sum(Y.^2,2);

% inner products between the two datasets
inner_prods = X * Y';

% add the three terms of the L2 distance (by expanding the ||x-y||^2_2 =
% ||x||^2_2 + ||y||^2_2 - 2x'*y
dist = bsxfun(@plus,norm_y',bsxfun(@minus,norm_x,2*inner_prods));

end