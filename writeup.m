function writeup()
% usage:
%  writeup()
% 
%  assumes all required files for writeup exist in the same directory as the
%  script

format compact;
close all;

% reads in csv files for training, validation, and test datasets
function [labels, data, imgs] = dataset(csv, i, j)
    num_data = size(csv, 1);
    labels = csv(:, 1);
    data = csv(:, 2:end);
    imgs = zeros(i, j, 1, num_data);
    for idx = 1:num_data
        imgs(:,:,:,idx) = reshape(data(idx,:), i, j, 1)';
    end
end

% read in datasets
train_csv = readmatrix('Kannada-MNIST/train.csv');
[train_labels, train_data, train_imgs] = dataset(train_csv, 28, 28);
test_csv = readmatrix('Kannada-MNIST/test.csv');
[test_labels, test_data, test_imgs] = dataset(test_csv, 28, 28);
val_csv = readmatrix('Kannada-MNIST/Dig-MNIST.csv');
[val_labels, val_data, val_imgs] = dataset(val_csv, 28, 28);

    function make_imgs(imgs, name)
        fig = figure;
        fig.Position = [100, 100, 1000, 400];
        tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
        for x = 1:10
            nexttile;
            imagesc(imgs(:,:,1,x));
            title(sprintf("%d", x-1));
            axis off;
        end
        saveas(gcf, sprintf("%s.png", name));
    end

    function [ten_imgs] = ten_imgs(data, i, j)
        ten_imgs = zeros(i, j,1, 10);
        for idx = 1:10
            ten_imgs(:,:,:,idx) = reshape(data(idx,:), i, j,1)';
        end
    end

if false
    make_imgs(ten_imgs(train_data, 28, 28), 'digits');
end

% normalization

train_data_N = normalize_imgs(train_data);
val_data_N = normalize_imgs(val_data);
test_data_N = normalize_imgs(test_data);

    function [imgs_norm] = normalize_imgs(imgs)
        % imgs_norm = zscore(imgs);
        imgs_norm = (imgs-min(imgs(:))) / (max(imgs(:)) - min(imgs(:)));
    end

% train_imgs_N = normalize_imgs(train_imgs);
% val_imgs_N = normalize_imgs(val_imgs);
% test_imgs_N = normalize_imgs(test_imgs);
train_imgs_N = normalize_imgs(train_imgs);
val_imgs_N = normalize_imgs(val_imgs);
test_imgs_N = normalize_imgs(test_imgs);

if false
    make_imgs(ten_imgs(train_data_N, 28, 28), 'normalized');
end

% principle component analysis

    function [s, num_pcs] = do_pca(data, make_chart_flag, chart_name)
        [coeff, score, latent] = pca(data);
        explained_variance = latent /sum(latent) * 100;
        if make_chart_flag
            figure;
            pareto(explained_variance);
            xlabel('Principal Component');
            ylabel('Variance Explained (%)');
            title('Variance Explained (%) versus Principal Component');
            saveas(gcf, sprintf('%s.png', chart_name));
        end
        cumulative_variance = cumsum(explained_variance);
        num_pcs = find(cumulative_variance >= 95, 1);
        fprintf('Number of PCs explaining 95%% of variance: %d\n', num_pcs);
        s = score(:,1:num_pcs);
    end
    
    function [s] = get_pca(data, num_pcs)
        [~, s, ~] = pca(data);
        s = s(:, 1:num_pcs);
    end

if false
[train_data_NP, tdNP_num_pcs] = do_pca(train_data_N, false, 'pca');
% val_data_NP = do_pca(val_data_N, false, '');
% test_data_NP = do_pca(test_data_N, false, '');
val_data_NP = get_pca(val_data_N, tdNP_num_pcs);
test_data_NP = get_pca(test_data_N, tdNP_num_pcs);
end

% canny edge detection

    function [edges] = canny_edge_detection(data, i, j)
        edges = zeros(size(data));
        for idx = 1:size(edges,1)
            edges(idx,:) = reshape(edge(reshape(data(idx,:),i,j)', 'canny')', 1, size(data,2));
        end
        
    end

if false
train_data_NC = canny_edge_detection(train_data_N, 28, 28);
val_data_NC = canny_edge_detection(val_data_N, 28, 28);
test_data_NC = canny_edge_detection(test_data_N, 28, 28);
end

if false
    make_imgs(ten_imgs(train_data_NC, 28, 28), 'edge');
end

% Canny edge into PCA

if false
[train_data_NCP, tdNCP_num_pcs] = do_pca(train_data_NC, false,'canny_edge_into_pca');
val_data_NCP = get_pca(val_data_N, tdNCP_num_pcs);
test_data_NCP = get_pca(test_data_N, tdNCP_num_pcs);
end

% data augmentation

iDA = imageDataAugmenter('RandRotation', [-45, 45], 'RandXScale', [0.75, 1.25],'RandYScale', [0.75, 1.25], 'RandXTranslation', [-2, 2], 'RandYTranslation', [-2, 2]);
    function [aug_imgs, aug_labels] = augment_img_data(imgs, labels)
        [l, m, n, o] = size(imgs);
        aug_imgs = zeros(l, m, n, 1*o);
        [x,y] = size(labels);
        aug_labels = zeros(x*1, y);
        for idx = 1:size(imgs, 4)
            aug_imgs(:,:,:,idx) = augment(iDA, imgs(:,:,:,mod(idx-1, o)+1));
            aug_labels(idx,:) = labels(mod(idx-1,x)+1, :);
        end
        aug_labels = categorical(aug_labels);
    end

[aug_imgs, aug_labels]= augment_img_data(train_imgs_N, train_labels);
aug_imgs = cat(4, aug_imgs, train_imgs_N);
aug_labels = cat(1, aug_labels, categorical(train_labels));

size(aug_imgs)
size(aug_labels)
aug_labels(60001)

if false
    make_imgs(aug_imgs(:,:,:,1:10), 'aug1');
end

% knn

    function [optimal_k, val_acc, time] = best_k(data, labels, val, val_labels)
        k_values = 3:2:7;
        val_acc = zeros(size(k_values));
        time = zeros(size(k_values));
        tic;
        for i = 1:length(k_values)
            Mdl = fitcknn(data, labels, 'NumNeighbors', k_values(i));
            pred = predict(Mdl, val);
            time(i) = toc
            val_acc(i) = sum(pred == val_labels) / size(val_labels,1) * 100
        end
        [~, optimal_k_idx] = max(val_acc);
        optimal_k = k_values(optimal_k_idx);
    end

if false
best_k(train_data, train_labels, val_data, val_labels);
best_k(train_data_N, train_labels, val_data_N, val_labels);
best_k(train_data_NC, train_labels, val_data_NC, val_labels);
best_k(train_data_NP, train_labels, val_data_NP, val_labels);
best_k(train_data_NCP, train_labels, val_data_NCP, val_labels);
end

% svm

    % function [val_acc, t] = do_svm(data, labels, val, val_labels)
    %     tic;
    %     % SVMModel = fitcsvm(data, labels, 'KernelFunction', 'linear');
    %     SVMModel = fitcecoc(data, labels, 'Learners', 'svm', 'Coding', 'onevsone');
    %     pred = predict(SVMModel, val);
    %     t = toc
    %     val_acc = sum(pred == val_labels) / size(val_labels,1) * 100
    % end

if false
% do_svm(train_data, train_labels, val_data, val_labels);
% do_svm(train_data_N, train_labels, val_data_N, val_labels);
% do_svm(train_data_NP, train_labels, val_data_NP, val_labels);
% do_svm(train_data_NCP, train_labels, val_data_NCP, val_labels);
end

% CNN

cat_train = categorical(train_labels);
cat_val = categorical(val_labels);
cat_test = categorical(test_labels);

layers = [
    imageInputLayer([28, 28, 1], 'Name', 'input', 'Normalization', 'none')  % Fixed size input

    % First convolutional layer
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')  % Max pooling layer
    
    % Second convolutional layer
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')  % Max pooling layer
    
    % Third convolutional layer
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2)

    % Fourth convolutional layer
    convolution2dLayer(3, 64, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Fully connected layer
    fullyConnectedLayer(10, 'Name', 'fc')  % 10 classes (digits 0–9)
    softmaxLayer('Name', 'softmax')  % Softmax output layer
    classificationLayer('Name', 'classoutput')  % Classification output layer
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...          % Number of epochs
    'MiniBatchSize', 128, ...    % Mini-batch size
    'InitialLearnRate', 0.001, ...
    'ValidationData', {val_imgs_N, cat_val}, ...  % Validation data
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');  % Plot training progress

if false
net = trainNetwork(train_imgs_N, cat_train, layers, options);
% pred_N = classify(net, val_imgs_N);
% accuracy = sum(pred_N == cat_val) / numel(cat_val);
% fprintf('Test Accuracy with Augmentation: %.2f%%\n', accuracy * 100);
% writematrix(pred_N, 'submission.csv');
end

% CNN with augmented data
if true
net = trainNetwork(aug_imgs, aug_labels, layers, options);
end

end