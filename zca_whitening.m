%%
%%
%%
%%
%%
function [data_zca, ZCA, mean_data] = zca_whitening(data, epsilon)

% compute the mean 
mean_data = mean(data);

% subtract the mean
data = bsxfun(@minus,data,mean_data);

% do SVD for PCA
C = cov(data);
[V,D] = eig(C);

% do ZCA whitening
ZCA = V * diag(1 ./ sqrt(diag(D) + epsilon)) * V';
data_zca = data * ZCA;

end