function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;
% Find indices of Admitted and NotAdmitted
admitted = find(y == 1);
not_admitted = find(y == 0);
% Plot training dataset
plot(X(admitted, 1), X(admitted, 2), 'k+', 'MarkerSize', 3, 'LineWidth', 2);
plot(X(not_admitted, 1), X(not_admitted, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 5);

hold off;

end
