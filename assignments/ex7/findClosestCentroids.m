function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set n
m = size(X, 1);

% Set K
K = size(centroids, 1);

% Return the closest centroids in idx for a dataset X
idx = zeros(m, 1);
for i = 1:m
  min = sum((X(i, :) - centroids(1, :)) .^ 2);
  min_idx = 1;
  for k = 1:K
    xith_to_centroidkth_dist = sum((X(i, :) - centroids(k, :)) .^ 2);
    if (xith_to_centroidkth_dist < min)
      min_idx = k;
      min = xith_to_centroidkth_dist;
    endif
  end
  idx(i) = min_idx;
end

end
