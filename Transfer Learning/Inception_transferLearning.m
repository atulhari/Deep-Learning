
rootfolder = 'C:\Users\Public\Documents\MATLAB';
datafolder = [rootfolder '\UTwente\ICH_data\ICH_data'];
 addpath(datafolder);
 load Labels.mat
 imds = imageDatastore(fullfile(datafolder,'im_ICH*'),'LabelSource','none');
imds.Labels=Labels;



close all % close all figure windows
clear variables % clear the workspace
%% 
load ICH_slice.mat
imshow(im1);
imgSize = size(im1);
imgSize = imgSize(1:2);

%% 
load trainednet_ex4.mat

features = activations(trainednet_ex4,im1,'conv_3','OutputAs','channels');
Rfeatures = reshape(features,[64 64 1 32]);
Feature_gray=mat2gray(Rfeatures);
montage(Feature_gray, 'BorderSize', 2, 'BackgroundColor', 'r');
print -r600 -dpng 1_9w.png


%% 


[maxValue,maxValueIndex] = max(max(max(Feature_gray)));
act1chMax = Feature_gray(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
imshow(act1chMax);
print -r600 -dpng 1_9o.png


%%
Figure1 = figure;
ax1 = axes('Parent', Figure1);
ax2 = axes('Parent', Figure1);
set(ax1, 'Visible', 'off');
set(ax2, 'Visible', 'off');
alpha = 0.6; % change this if appropriate
I = imshow(act1chMax, 'Parent', ax2, 'colormap', jet);
set (I, 'AlphaData', alpha);
imshow(im1, 'Parent', ax1);
print -r600 -dpng 1_3q.png

%% 
features2 = activations(trainednet_ex4,im1,'relu_3','OutputAs','channels');
Rfeatures2 = reshape(features2,[64 64 1 32]);
Feature_gray2=mat2gray(Rfeatures2);
montage(Feature_gray2, 'BorderSize', 2, 'BackgroundColor', 'r');
print -r600 -dpng 2_2.png




figure;
[maxValue2,maxValueIndex2] = max(max(max(Feature_gray2)));
act1chMax2 = Feature_gray2(:,:,:,maxValueIndex2);
act1chMax2 = mat2gray(act1chMax2);
act1chMax2 = imresize(act1chMax2,imgSize);
imshow(act1chMax2);
print -r600 -dpng 2_3.png



Figure2 = figure;
ax1 = axes('Parent', Figure2);
ax2 = axes('Parent', Figure2);
set(ax1, 'Visible', 'off');
set(ax2, 'Visible', 'off');
alpha2 = 0.6; % change this if appropriate
I = imshow(act1chMax2, 'Parent', ax2, 'colormap', jet);
set (I, 'AlphaData', alpha2);
imshow(im1, 'Parent', ax1);
print -r600 -dpng 2_8png


features2 = activations(trainednet_ex4,im1,'relu_3','OutputAs','channels');
Rfeatures2 = reshape(features2,[64 64 1 32]);
Feature_gray2=mat2gray(Rfeatures2);
montage(Feature_gray2, 'BorderSize', 2, 'BackgroundColor', 'r');
print -r600 -dpng 2_2q.png




figure;
[maxValue2,maxValueIndex2] = max(max(max(Feature_gray2)));
act1chMax2 = Feature_gray2(:,:,:,maxValueIndex2);
act1chMax2 = mat2gray(act1chMax2);
act1chMax2 = imresize(act1chMax2,imgSize);
imshow(act1chMax2);
print -r600 -dpng 2_3q.png



Figure2 = figure;
ax1 = axes('Parent', Figure2);
ax2 = axes('Parent', Figure2);
set(ax1, 'Visible', 'off');
set(ax2, 'Visible', 'off');
alpha2 = 0.6; % change this if appropriate
I = imshow(act1chMax2, 'Parent', ax2, 'colormap', jet);
set (I, 'AlphaData', alpha2);
imshow(im1, 'Parent', ax1);
print -r600 -dpng 2_4q.png

%% 

features2 = activations(trainednet_ex4,im2,'relu_3','OutputAs','channels');
Rfeatures2 = reshape(features2,[64 64 1 32]);
Feature_gray2=mat2gray(Rfeatures2);
montage(Feature_gray2, 'BorderSize', 2, 'BackgroundColor', 'r');
print -r600 -dpng 2_2a.png




figure;
[maxValue2,maxValueIndex2] = max(max(max(Feature_gray2)));
act1chMax2 = Feature_gray2(:,:,:,maxValueIndex2);
act1chMax2 = mat2gray(act1chMax2);
act1chMax2 = imresize(act1chMax2,imgSize);
imshow(act1chMax2);
print -r600 -dpng 2_3a.png



Figure2 = figure;
ax1 = axes('Parent', Figure2);
ax2 = axes('Parent', Figure2);
set(ax1, 'Visible', 'off');
set(ax2, 'Visible', 'off');
alpha2 = 0.6; % change this if appropriate
I = imshow(act1chMax2, 'Parent', ax2, 'colormap', jet);
set (I, 'AlphaData', alpha2);
imshow(im2, 'Parent', ax1);
print -r600 -dpng 2_4a.png


%% 
close all;

  ReluIM(im1,1);  % see the function definition below

   ReluIM(im2,2);
  
    ReluIM(im4,3);
  
     ReluIM(im4,4);

%% 
% Using the inception deep learning network
net=inceptionv3;
layers=net.Layers;

%% 
connections = net.Connections;
layers(2)=[];
layers(315)=[];
layers(314)=[];
layers(313)=[];


layers(1)=imageInputLayer([128 128 1],'Name','input_1');
layers(2)=convolution2dLayer(3,32,'Stride',2,'NumChannels',1,'Name','conv2d_1');
layers(312)=averagePooling2dLayer([2 2],'Stride',[2 2],'Name','avg_pool');
layers(313)=fullyConnectedLayer(2,'Name','predictions');
layers(314)=softmaxLayer('Name','predictions_softmax');
layers(315)=classificationLayer('Name','ClassificationLayer_predictions');
connections(2,:)=[];
connections.Destination{1}=connections.Source{2};
net = createLgraphUsingConnections(layers,connections);

%%
options = trainingOptions('adam',...
    'MaxEpochs',100,...
    'ValidationData', valset,...
    'ValidationPatience',10,...
    'ValidationFrequency',11,...
    'Verbose',true,...
    'LearnRateSchedule','piecewise',...
    'Shuffle','every-epoch',...
    'MiniBatchSize', 128,...
    'Plots','training-progress');


netnew=trainNetwork(trainset,net,options);
%%
trainpred=classify(netnew,trainset);
labeltrain=trainset.Labels;
confusiontrain=confusionmat(labeltrain,trainpred);

valpred=classify(netnew,valset);
labelval=valset.Labels;
confusionval=confusionmat(labelval,valpred);

testpred=classify(netnew,testset);
labeltest=testset.Labels;
confusiontest=confusionmat(labeltest,testpred);


function ReluIM(b,i)
 rootfolder = 'C:\Users\Public\Documents\MATLAB';
 load ICH_slice.mat
 load trainednet_ex4.mat

name=['save',num2str(i)];

features3 = activations(trainednet_ex4,b,'relu_3','OutputAs','channels');
 sizeim=size(features3);
   % sizeim=sizeim(1:2);
Rfeatures3 = reshape(features3,[sizeim(1) sizeim(2) 1 sizeim(3)]);
Feature_gray3=mat2gray(Rfeatures3);
montage(Feature_gray3, 'BorderSize', 2, 'BackgroundColor', 'r');
figure;
print(name,'-dpng') ;


figure;
[maxValue3,maxValueIndex3] = max(max(max(Feature_gray3)));
act1chMax3 = Feature_gray3(:,:,:,maxValueIndex3);
act1chMax3 = mat2gray(act1chMax3);
act1chMax3 = imresize(act1chMax3,size(b));
imshow(act1chMax3);
name=['save1',num2str(i)];
print(name,'-dpng') ;

Figure3 = figure;
ax1 = axes('Parent', Figure3);
ax2 = axes('Parent', Figure3);
set(ax1, 'Visible', 'off');
set(ax2, 'Visible', 'off');
alpha3 = 0.6; % change this if appropriate
I = imshow(act1chMax3, 'Parent', ax2, 'colormap', jet);
set (I, 'AlphaData', alpha3);
imshow(b, 'Parent', ax1);
name=['save2',num2str(i)];
print(name,'-dpng') ;


end

