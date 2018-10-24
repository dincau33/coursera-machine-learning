function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Calculate value of the hidden layer
a2 = sigmoid(X * (Theta1)');

% Add ones to the X data matrix
a2 = [ones(size(a2,1), 1) a2];

% Calculate value of the output layer
a3 = sigmoid(a2 * (Theta2)');

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
[~, p] = max(a3, [], 2);

end
