function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

c_test   = [.01,.03,.1,.3,1,3,10,30];
sig_test = c_test;
num_test = length(c_test);
err = zeros(num_test,num_test);
for i = 1:num_test
  for j = 1:num_test
    model= svmTrain(X, y, c_test(i), @(x1, x2) gaussianKernel(x1, x2, sig_test(j))); 
    predictions = svmPredict(model, Xval);
    err(i,j) = mean(double(predictions ~= yval));
  end
end

minVal = C(1,1);
C = c_test(1);
sigma = sig_test(1);

for i = 1:num_test
  for j = 1:num_test
    if(err(i,j) < minVal)
      C = c_test(i);
      sigma = sig_test(j);
      minVal = err(i,j);
    end
  end
end

a = 1;

% =========================================================================

end
