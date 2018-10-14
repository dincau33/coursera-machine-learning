function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1
[cost, grad] = costFunction(theta, X, y);

%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
J = cost + (lambda / (2 * m)) * (theta(2:n))' * theta(2:n);
grad = grad + (lambda / m) * [0;theta(2:n)];

end
