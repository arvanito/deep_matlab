%%  data = contrast_normalization(data, epsilon):
%%
%%  function that normalizes the data to have zero mean and 
%%  unit variance. It operates as contrast normalization.
%%
%%  Input:
%%      data: Initial data points, each row represents one data point
%%      epsilon: regularizer for division with standard deviation
%%
%%  Output:
%%      std_data: contrast normalized data
%%
function std_data = contrast_normalization(data, epsilon)

% subtract the mean from each data point
data = bsxfun(@minus,data,mean(data,2));

% divide by standard deviation by adding a regularizer 
std_data = bsxfun(@rdivide,data,sqrt(var(data,[],2)+epsilon));

end