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
                 hidden_layer_size, (input_layer_size + 1)); % 25 * 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 * 26

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 * 401
Theta2_grad = zeros(size(Theta2)); % 10 * 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X = [ones(m, 1) X];

y_new = ones(m, num_labels);
for line = 1: m
    y_new(line, :) = 1: num_labels;
end

y_new = (y_new == y);
size(y_new); % 5000 * 10

% 每一层都要sigmoid计算
z_2 = [ones(m, 1) X * Theta1'];
hidden_layer = [ones(m, 1) sigmoid(X * Theta1')]; % 5000 * 26
output_layer = hidden_layer * Theta2'; % 5000 * 10;
h =  sigmoid(output_layer); % 5000 * 10;

% ====================== J without regularization ======================
%% 方法 1：without regularization
% for line=1:m
%     J += -(y_new(line,:) * log(h(line,:))' + (1 - y_new(line,:)) * log(1 - h(line,:))');
% end
% J = J / m;


%% 方法 2：wiithout regularization
% y = y_new;
% j_matrix = -(y .* log(h) + (1 - y) .* log(1 - h));
% J = sum(sum(j_matrix'))/m;


% ====================== J with regularization ======================
y = y_new;
j_matrix = -(y .* log(h) + (1 - y) .* log(1 - h));
J_1 = sum(sum(j_matrix')) / m;

reg_matrix = sum(sum(Theta1(:,[2:end]) .^ 2)) + sum(sum(Theta2(:,[2:end]) .^ 2));
J_2 = reg_matrix * lambda / (2*m);
J = J_1 + J_2;

% ======================    backpropagation    ======================
delta_3 = h - y_new; % 5000 * 10
% fprintf('size of delta3 %d %d \n' , size(delta_3));
delta_2 = delta_3 * Theta2 .* sigmoidGradient(z_2); % 5000 * 26
% fprintf('size of delta2 %d %d \n' , size(delta_2));
Dt_1 = delta_2(:, 2:end)' * X; % 25 * 401
% fprintf('size of Dt_1   %d %d \n' , size(Dt_1));
Dt_2 = delta_3' * hidden_layer; % 10 * 26
% fprintf('size of Dt_2   %d %d \n' , size(Dt_2));
Theta1_grad = Dt_1 / m;
Theta2_grad = Dt_2 / m;


% ============   backpropagation with regularization    ===========

Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end); 







% Dt_1 = zeros(hidden_layer_size, input_layer_size + 1); % 25 * 401
% Dt_2 = zeros(num_labels, hidden_layer_size + 1); % 10 * 26
% for c = 1: m
%     delta_3 = h(c,:)' - y_new(c,:)'; % 10 * 1
%     delta_2 = Theta2' * delta_3 .* sigmoidGradient(hidden_layer(c, :)'); % 26 * 1
%     Dt_1 += delta_2(2:end) * X(c, :);
%     Dt_2 += delta_3 * hidden_layer(c,:);
%
% end
%
% Theta1_grad = Dt_1 / m;
% Theta2_grad = Dt_2 / m;
% --------------------  regularized part ------------------------







% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
