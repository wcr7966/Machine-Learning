function [J, grad] = Cost_LR(w, x, y)
m = length(y);
h = Sigmoid(x*w);
J = -1/m .* sum(y.*log(h) + (1-y).*log(1-h));
grad = (x' * (h - y))/m;