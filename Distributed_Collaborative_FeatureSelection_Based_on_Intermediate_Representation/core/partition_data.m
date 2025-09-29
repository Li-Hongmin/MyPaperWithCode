function Div_data = partition_data(X_train, labels_train, num_divisions)
%PARTITION_DATA Partition training data into distributed divisions
%
% Input:
%   X_train - Training data matrix (features × samples)
%   labels_train - Training labels vector
%   num_divisions     - Number of distributed divisions
%
% Output:
%   Div_data - Structure array containing partitioned data

num_train_samples = size(X_train, 2);

% Initialize division data structure
Div_data = struct('X', {}, 'l', {});

for division_idx = 1:num_divisions
    start_idx = (division_idx-1)*num_train_samples/num_divisions + 1;
    end_idx = division_idx*num_train_samples/num_divisions;

    Div_data(division_idx).X = X_train(:, start_idx:end_idx);
    Div_data(division_idx).l = labels_train(start_idx:end_idx);
end

end