function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X); % Computes mean value of each feature
sigma = std(X); % Computes standard deviation of each feature
t = ones(size(X,1),1);
X_norm = (X - (t * mu)) ./ (t * sigma); % Computes a normalized version of X

end
