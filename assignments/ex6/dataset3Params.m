function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% List of values to try for C and sigma
C_val = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_val = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Initialization
C = 0;
sigma = 0;
best_prediction_error = 10000;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

for i = 1:size(C_val)
  for j = 1:size(sigma_val)
    % Train kernel
    model = svmTrain(X, y, C_val(i), @(x1, x2) gaussianKernel(x1, x2, sigma_val(j)));
    % Compute prediction error
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    % Assess
    if prediction_error < best_prediction_error
        best_prediction_error = prediction_error;
        C = C_val(i);
        sigma = sigma_val(j);
    end
    fprintf(['C_val = %f - sigma_val = %f\n'], C_val(i), sigma_val(j));
    C
    sigma
  end
end

end
