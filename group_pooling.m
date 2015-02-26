%%  pooled_features = group_pooling(features, q1, q2, D, groups, num_groups)
%%
%%  function that does max-pooling on learned filters. The goal is to 
%%  reduce the number of filters by maintaining learned information.
%%
%%  Input:
%%      features: input features to be max-pooled relative to learned filters
%%      q1: first dimension of input features
%%      q2: second dimension of input features
%%      D: learned filters
%%      groups: max-pooling groups for the learned filters
%%      num_groups: number of groups to pool over
%%
%%  Output:
%%      pooled_features: final max-pooled features
%% TODO:: extend it to many image features
function pooled_features = group_pooling(features, q1, q2, D, groups, num_groups)

% reshape features to 2-D matrix, columns correpond to features
num_features = size(D,1);
features = reshape(features,q1*q2,num_features);

% allocate memory for the pooled features
pooled_features = zeros(q1*q2,num_groups);

% max pool the feature responses over the learned filter groups
for q = 1:q1*q2
    for g = 1:num_groups
        % find current group of filters
        D_g = (groups==g);
        pooled_features(q,g) = max(features(q,D_g));
    end
end

pooled_features = pooled_features(:)';

end