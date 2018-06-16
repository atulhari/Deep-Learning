%% Linear and quadratic classifier 

close all % close all figure windows
clear variables % clear the workspace
%% Loading dataset
load anthropometry
prwarning off % set the prtools warning level to 'off'
prwaitbar off % set the prtools waitbar level to 'off'
gridsize(200); % set the prtools gridsize for plot commands to ‘fine’
fd = prdataset(wh_female,'female'); % create the data set fd (women)
md = prdataset(wh_male,'male'); % create the data set md (men)

%% 
fm=[fd;md];
fm.featlab = ['weight (kg)';'height (cm)']; % add names to measurements
figure;
scatterd(fm,'legend','.'); % create a scatter diagram
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
print -r600 -dpng scatter1.png % create a png file with the diagram

%% 
% 
Fi = 135; % see ex 2.3 % estimated angle of perpendicular line (degrees)
w = [cosd(Fi); sind(Fi)]; % weights
g_female = wh_female*w; % linear discriminant women
g_male = wh_male*w; % linear discriminant men
figure;
plot(g_female,g_female*0,'b+',g_male,g_male*0,'r*');
xlabel('projected data');
legend('female','male'); % add a legend
print -r600 -dpng projected_data.png % create a png file with the diagram


%% 
Nf = length(g_female); % number of females
Nm = length(g_male); % number of males
T = [66 67 68 69 70]; % threshold (choose five of them)
for i=1:length(T)
E(1,i) = (sum(g_female>T(i)) + sum(g_male<=T(i)))/(Nf+Nm); % error rate
E(2,i) = sum(g_female>T(i))/Nf; % error rate type I
E(3,i) = sum(g_male<=T(i))/Nm; % error rate type II
end
disp(E)

%% 

clear variables
close all
load brain % load brain dataset
figure (6);
imshowpair(mFLAIR,mT1,'montage') % show both MRI images
brain = prdataset(learnset.data,learnset.labels);
figure(7);
scatterd(brain, 'legend','.');
print -r600 -dpng scatter2.png % create a png file with the diagram
%% 
wldc = ldc(brain); % create a Linear Bayes Normal Classifier
figure(8);
scatterd(brain,'.'); % create a scatter diagram
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
plotm(wldc,2); % plot the contour lines of the gaussians
print -r600 -dpng 3_2.png
%% 
figure (9);
scatterd(brain,'.'); % create a scatter diagram
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
plotc(wldc,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng 3_3.png

%% Q 
figure (10);
scatterd(brain*wldc,3); % calculate the soft labels and show them
axis equal % make sure that axis scales are equal
title('soft labels')
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
print -r600 -dpng 3_5.png

%% 
error_rate = brain*wldc*testc;%% Calculating error rate
%% 
im = cat(3,mFLAIR,mT1); % concatenate the two images
imset = im2feat(im); % convert it to prtools data set
labels = imset*wldc*labeld; % classify every pixel in the image
labels(~mask) = 0; % use mask to remove background
im = reshape(labels,256,256); % reshape image
figure;
imagesc(im) % visualize classification results
print -r600 -dpng 3_7.png

%% 
close all
wqdc = qdc(brain); % create a quadratic Bayes Normal Classifier
figure(8);
scatterd(brain,'.'); % create a scatter diagram
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
plotm(wqdc,2); % plot the contour lines of the gaussians
print -r600 -dpng 3_8.png
%%
figure (9);
scatterd(brain,'.'); % create a scatter diagram
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
plotc(wqdc,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng 3_38b.png

%%
figure (10);
scatterd(brain*wqdc,3); % calculate the soft labels and show them
axis equal % make sure that axis scales are equal
title('soft labels')
legend('CSF', 'WM', 'GM'); % create legend for scatterplot
print -r600 -dpng 3_8c.png

%%
error_rate_q = brain*wqdc*testc;%% Calculating error rate
%%
im = cat(3,mFLAIR,mT1); % concatenate the two images
imset = im2feat(im); % convert it to prtools data set
labels = imset*wqdc*labeld; % classify every pixel in the image
labels(~mask) = 0; % use mask to remove background
im = reshape(labels,256,256); % reshape image
figure;
imagesc(im) % visualize classification results
print -r600 -dpng 3_8d.png


