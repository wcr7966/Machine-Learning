train_feature = "./assign2_dataset/page_blocks_train_feature.txt";
train_lable = "./assign2_dataset/page_blocks_train_label.txt";
test_feature = "./assign2_dataset/page_blocks_test_feature.txt";

%LR(); %二分类对数几率回归
X = load(train_feature);
Y = load(train_lable);
T = load(test_feature);
num_label = 5;
all_w = LR_OVR(X, Y, num_label, T);