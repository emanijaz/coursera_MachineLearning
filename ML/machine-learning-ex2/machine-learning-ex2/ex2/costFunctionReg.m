function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g=zeros(m,1);
for i=1:m,
   g(i) = sigmoid(X(i,:)*theta);
end;
for i=1:m,
   J += (-y(i)* log(g(i))) - ((1-y(i))*log(1-g(i)));
end;

J = J/m;

regTheta=0;
for i =2:size(theta(:,1)),
	regTheta += theta(i)^2;
end;
J += (regTheta*lambda)/(2*m);


der = 0;
for i=1:m,
     der += (g(i) - y(i))* X(i, 1);
end;
	der = der/m;
	grad(1) = der;
   
for  j=2:size(grad)(1),
   der = 0;
   for i=1:m,
     der += (g(i) - y(i))* X(i, j);
   end;
     der = der/m;
	 der += (lambda/m)* theta(j);
	 grad(j) = der;
end;




% =============================================================

end
