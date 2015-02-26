%%  f = compute_activation(data, D, activation_type):
%%  
%%  function that computes the activation for the feature extraction step.
%%
%%  Input:
%%      data: data points, rows correspond to points, columns to features
%%      D: learned centroids from the feature learning procedure
%%      activation_type: type of activation used
%%
%%  Output:
%%      f: final activations of the data points
%%
%%  TODO:: More activation functions!!!! DEBUG THEM!!!!
function f = compute_activation(data, D, activation_type, varargin)

if (strcmp(activation_type,'triangle')==1) % triangle activation function
    
    % compute distances between data and learned centroids
    dist = compute_distances(data, D);    
    
    % average distance to centroids for each patch
    mu = mean(dist,2);     
    
    % compute thresholded activation
    f = max(bsxfun(@minus,mu,dist),0);
elseif (strcmp(activation_type,'threshold')==1)
    
    % extract features by multiplying the input patches with the centroids
    z = data * D';
    
    % compute thresholded features
    if (isempty(varargin))
        alpha = 0;
    else
        alpha = varargin{1};
    end
    %f = [max(z-alpha,0), max(-z-alpha,0)];
    f = max(z-alpha,0);
elseif (strcmp(activation_type,'abs')==1)
    
    % extract features by multiplying the input patches with the centroids
    z = data * D';
    
    % compute absolute-valued features
    %f = abs(z);
    f = z;
end

end