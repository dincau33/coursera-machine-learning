function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Calculate value of the hidden layer
a2 = (sigmoid(Theta1 * (X)'))';

% Add ones to the X data matrix
a2 = [ones(m, 1) a2];

% Calculate value of the output layer
a3 = (sigmoid(Theta2 * (a2)'))';

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
[~, p] = max(a3, [], 2);

end
