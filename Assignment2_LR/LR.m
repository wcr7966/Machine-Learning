function [] = LR()
datafile = './assign2_dataset/data.csv';
targetfile = './assign2_dataset/targets.csv';
D = csvread(datafile);
T = csvread(targetfile);
X = D(1: 50, :);
Y = T(1: 50, :);
test = D(51: 80, :);
grid on
hold on

%归一化
X2 = mapminmax(X');
X = X2';

[m, n] = size(X);
X = [ones(m, 1) X]; %在列向量左边添加全1列向量
w = ones(n + 1, 1); %初始化w
w_new = zeros(n + 1, 1);
step = 0.5;
%[cost, grad] = Cost_LR(w, X, Y);
while (w_new - w) * (w_new - w)' > 0.00000001
    w = w_new;
    [J_pre, grad] = Cost_LR(w, X, Y);
    w_new = w_new - step * grad;
end
%disp(w_new);
 
w_new = w_new(1:n, :);
[m2, ~] = size(test);

z = test(:, 1:n)*w_new(1:n, :);
%归一化
z2 = z';
z2 = mapminmax(z2, -1, 1);
y = Sigmoid(z2');

for i = 1: m2
    if(y(i, 1) >= 0.5)
        y(i, 1) = 1;
    else
        y(i, 1) = 0;
    end
end
plot(test, y, 'ro');

count = 0.0;
for i = 51: 80
    if T(i, :) == y(i - 50, 1)
        count = count + 1;
    end
end

disp(count);
disp(count/30.0);

csvwrite('A.csv', y);