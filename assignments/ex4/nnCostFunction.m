function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % (25 * 401)

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % (10 * 26)

% Setup some useful variables
m = size(X, 1);

% Calculate hypothesis by applying feedforward algorithm
% Add ones to the X data matrix
X = [ones(m, 1) X];
% Calculate activation outputs of the hidden layer
z2 = X * (Theta1)'; % (5000 * 25)
a2 = sigmoid(z2); % (5000 * 25)
% Add bias to the activation outputs
a2 = [ones(size(a2,1), 1) a2];
% Calculate activation outputs of the output layer
z3 = a2 * (Theta2)';
a3 = sigmoid(z3);
% Calulate hypothesis
H = a3; %(5000 * 10)

% Vectorize y as y_vec
for i = 1:m
   y_vec(i, y(i)) = 1; %(5000 * 10)
end

% Cost function without regularization
J_no_reg = (1 / m) * sum(sum((-y_vec) .* log(H) - (1 - y_vec) .* log(1 - H)));

% Cost function with regularization
Theta1_reg = Theta1(:, 2:size(Theta1,2));
Theta2_reg = Theta2(:, 2:size(Theta2,2));
J_reg = (lambda / (2 * m)) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));
J = J_no_reg + J_reg;

% You need to return the following variables correctly
%Theta1_grad_no_reg = zeros(size(Theta1));
%Theta2_grad_no_reg = zeros(size(Theta2));

% Backpropagation algorithm without ragularization
%for t = 1:m
   % Step 1: feedforward
   % Step 2: error calculation for output layer
%   delta3 = ((a3(t,:))' - (y_vec(t,:))'); % (10 * 1)
   % Step 3: error calculation for hidden layer
%   delta2 =  ((Theta2)' * delta3) .* [1; (sigmoidGradient(z2(t,:)))']; % (26 * 1)
%   delta2 = delta2(2:end); % (25 * 1)
   % Step 4:
%   Theta1_grad_no_reg = Theta1_grad_no_reg + delta2 * (X(t,:)); %(25 * 400)
%   Theta2_grad_no_reg = Theta2_grad_no_reg + delta3 * (a2(t,:)); %(10 * 25)
%end

% Backpropagation algorithm without ragularization
% Step 1: feedforward
% Step 2: error calculation for output layer
delta3 = a3 - y_vec; % (5000 * 10)
% Step 3: error calculation for hidden layer
delta2 = (delta3 * Theta2) .* [ones(m, 1) sigmoidGradient(z2)]; % (5000 * 26)
delta2 = delta2(:, 2:end);
% Step 4:
Theta1_grad_no_reg = (1/m) * (delta2)' * X; %(25 * 400)
Theta2_grad_no_reg = (1/m) * (delta3)' * a2; %(10 * 25)

%Theta1_grad_no_reg = (1/m) * Theta1_grad_no_reg;
%Theta2_grad_no_reg = (1/m) * Theta2_grad_no_reg;

% Backpropagation algorithm without ragularization
Theta1_grad_reg = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad_reg = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad_no_reg + Theta1_grad_reg;
Theta2_grad = Theta2_grad_no_reg + Theta2_grad_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
