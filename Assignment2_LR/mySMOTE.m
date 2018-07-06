function [rx, ry] = mySMOTE(X, Y, k, index)
[m, n] = size(X);
Y2 = Y;

% pos是多类，neg是少类
this = 0;
that = 0;
neg = [];
pos = [];
for j = 1: m
    if Y2(j, :) == index
        this = this + 1;
        neg(this, :) = X(j, :);
    else
        that = that + 1;
        pos(that, :) = X(j, :);
    end
end

minority = neg;

[m2, n2] = size(minority);
rx = [];
ry = [];

%对每一个少类中的样本进行处理
for sample_index = 1: m2
    distance = zeros(m2, 1);
    for i = 1: m2
        if i == sample_index
            distance(i) = inf;
        else
            distance(i) = dist(minority(sample_index, :), minority(i, :)');
        end
    end
    
    distance = [distance, neg];
    distance = sortrows(distance, 1);
    %删除排序列
    distance(:, 1) = [];
    
    for r = 1: k
        l = round(rand(1)*k)+1;
        l = distance(l, :);
        new_sample = zeros(size(l));
        for j = 1: n2
            gap = rand(1);
            dif = l(:, j) - minority(sample_index, j);
            new_sample(:, j) = minority(sample_index, j) + gap * dif;
            new_sample_y= index;
        end
        rx = [rx; new_sample];
        ry = [ry; new_sample_y];
    end
end