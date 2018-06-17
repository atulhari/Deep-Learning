

clc
clear variables
close all

%

rootfolder = cd; % get the current directory
datafolder = [rootfolder '\ICH_data\ICH_data']; % change this if needed
addpath(datafolder); % add this folder to the search path
load Labels.mat; % load image labels
imds = imageDatastore(fullfile(datafolder, 'im_ICH*'), 'LabelSource', 'none');
imds.Labels = Labels; % associate labels



figure; % open new figure
perm = randperm(length(imds.Files),20); % select 20 random rows
for i = 1:20 % for each of the 20 random row
    subplot(4,5,i); % create subplot 4 rows by 5 columns
    imshow(imds.Files{perm(i)}); % show random image from ImageDatastore
end
random_labels = Labels(perm)'; % show labels of the random images
countEachLabel(imds);
[trainset,valset,testset] = splitEachLabel(imds,0.6,0.2,0.2,'randomized');

%layers
layers = [imageInputLayer([128 128 1])
    convolution2dLayer(3,32,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,32,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,64,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,64,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,128,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];





options = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...
    'ValidationData',valset, ...
    'ValidationPatience', Inf, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Shuffle', 'every-epoch',...
    'Plots','training-progress');

net =trainNetwork(trainingset,layers,options); %training