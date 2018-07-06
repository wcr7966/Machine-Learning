function g = Sigmoid(z)
g = 1.0 ./ (1.0 + exp(-1.0.*z));