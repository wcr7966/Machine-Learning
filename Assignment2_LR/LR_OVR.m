function [all_w] = LR_OVR(X, Y, num_label, test)

%smote处理类别不平衡问题
[x2, y2] = mySMOTE(X, Y, 10, 2);
[x3, y3] = mySMOTE(X, Y, 20, 3);
[x4, y4] = mySMOTE(X, Y, 30, 4);
[x5, y5] = mySMOTE(X, Y, 15, 5);
X = [X; x2; x3; x4; x5];
Y = [Y; y2; y3; y4; y5];

[m, n] = size(X);
X = [ones(m, 1) X];

%归一化
X2 = mapminmax(X');
X = X2';

step = 5e-4;
all_w = ones(n+1, num_label);
for i = 1: num_label
    disp(i);
    w = ones(n + 1, 1); %初始化w
    w_new = zeros(n + 1, 1);
    Y2 = Y;
    for j = 1: m
        if Y2(j, 1) == i
            Y2(j, 1) = 1;
        else
            Y2(j, 1) = 0;
        end
    end
    while (w_new - w) * (w_new - w)' > 0.0000001
        w = w_new;
        [J_pre, grad] = Cost_LR(w, X, Y2);
        w_new = w_new - step * grad;
    end
    all_w(:, i) = w_new;
end

disp("--------test-------");
[m2, n2] = size(test);

%归一化
z = mapminmax(test');
z = z';
z = z *  all_w(1:n,:);
y = Sigmoid(z);

%对每个样本进行计算，选出最大的计算函数
for i = 1: m2
    max = y(i, 1);
    max_index = 1;
    for j = 2: num_label
        if max < y(i, j)
            max = y(i, j);
            max_index = j;
        end
    end
    y(i, 1) = max_index;   
end

T = load("./assign2_dataset/page_blocks_test_label.txt");
total = 0.0;
for c = 1: num_label
    count = 0.0;
    for j = 1: m2
        if (T(j, 1) == y(j, 1)) && (y(j, 1) == c)
            count = count + 1;
        end
        if T(j, 1) == y(j, 1)
            total = total + 1;
        end
    end
    acc1 = count/sum(T(:,1) == c);
    acc2 = count/sum(y(:, 1) == c);
    disp(c + "的查全率：" + acc1);
    disp(c + "的查准率" + acc2);
end
total = total/5;
disp(total);
disp(m2);
acc = total/m2;
disp("准确率: " + acc);

fp=fopen('A.txt','w');
fprintf(fp,'%d ',y);
fclose(fp);

figure;
plot(test,y(:, 1),'ro');
end

