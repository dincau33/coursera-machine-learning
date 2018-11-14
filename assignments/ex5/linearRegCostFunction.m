function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
H = X * theta; % Computes h for each x contained in training set
theta_reg = [0;theta(2:n)];

% Compute the cost and gradient of regularized linear
% regression for a particular choice of theta
J = (1 / (2 * m)) * (H - y)' * (H - y);
J = J + (lambda / (2 * m)) * (theta_reg)' * theta_reg;

grad = (1 / m) * ((H - y)' * X)';
grad = grad + (lambda / m) * theta_reg;
grad = grad(:);

end
