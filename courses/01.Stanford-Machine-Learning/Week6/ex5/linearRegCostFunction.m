function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate hypothesis
h_theta = X * theta;

% calculate mean square error
square_errors = (h_theta - y) .^ 2;

% regularization parameter
reg = (lambda / (2 * m)) * (theta(2:length(theta))' * theta(2:length(theta)));

% calculate cost function with regularization
J = (1 / (2 * m)) * sum(square_errors) + reg;


% =========================================================================

% calculate regularized linear regression gradient
grad = (1 / m) * ((h_theta - y)' * X);
grad(:, 2:length(grad)) += (lambda / m * theta(2:length(theta))');

grad = grad(:);

end
