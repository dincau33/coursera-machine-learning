function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% Returns the new centroids by
% computing the means of the data points assigned to each centroid
centroids = zeros(K, n); % (K * n)

for k = 1:K
  xk_idx = idx == k;
  Ck = sum(xk_idx);
  meank = (1 / Ck) * (xk_idx)' * X;
  centroids(k,:) = meank;
end

end
