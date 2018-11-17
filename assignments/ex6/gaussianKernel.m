function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Return the similarity between x1
% and x2 computed using a Gaussian kernel with bandwidth
% sigma
sim = exp(-(1 / (2 * sigma ^ 2)) * (x1 - x2)' * (x1 - x2));
end
