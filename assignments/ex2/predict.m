function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta

m = size(X, 1); % Number of training examples

% Compute probability of p=1 for x given theta
H = sigmoid(X * theta);

%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
p = zeros(m, 1);
p = H >= 0.5;

end
