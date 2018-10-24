function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
H = sigmoid(X * theta); % Computes h for each x contained in training set
% Cost function without regularization
J_no_reg = (1 / m) * ((-(y)' * log(H)) - ((1 - y)' * log(1 -  H)));
% Gradient of the cost function without regularization
grad_no_reg = (1 / m) * ((H - y)' * X)';

% Initialize some useful values
n = length(theta); % number of features + 1
theta_reg = [0;theta(2:n)];
% Cost function with regularization
J = J_no_reg + (lambda / (2 * m)) * (theta_reg)' * theta_reg;
% Gradient of the cost function with regularization
grad = grad_no_reg + (lambda / m) * theta_reg;
end
