function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


hypothesis = X * Theta';

% multiply with rating matrix, R so that Cost Function only
% accounts for movie i that has been rated by user j
% R{0, 1} and 1 if movie is rated else 0

% calculate total cost
J = 1/2 * sum(sum((R .* (hypothesis - Y)) .^2));

% calculate gradient for X (movie feature) and Theta (user weights)
X_grad = (R .* (hypothesis - Y)) * Theta;
Theta_grad = (R .* (hypothesis - Y))' * X;


% Perform regularization on Cost Function and Gradients

% calculate regularization parameters
reg_theta = lambda / 2 * (sum(sum(Theta .^ 2)));
reg_X = lambda / 2 * (sum(sum(X .^ 2)));

% regularize cost function
J = J + reg_theta + reg_X;

% regularize gradients
X_grad = X_grad + (lambda * X);
Theta_grad = Theta_grad + (lambda * Theta);


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
