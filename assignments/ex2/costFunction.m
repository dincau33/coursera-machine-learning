function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
H = sigmoid(X * theta); % Computes h for each x contained in training set
% Cost function
J = (1 / m) * ((-(y)' * log(H)) - ((1 - y)' * log(1 -  H)));
% Gradient of the cost function
grad = (1 / m) * ((H - y)' * X)';

end
