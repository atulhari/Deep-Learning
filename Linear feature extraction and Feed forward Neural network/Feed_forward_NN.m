
clc

close all % close all figure windows
clear variables % clear the workspace
%%
load ax_diff_data.mat
prwarning off % set the prtools warning level to 'off'
prwaitbar off % set the prtools waitbar level to 'off'
%gridsize(200); % set the prtools gridsize for plot commands to ‘fine’
trainset=prdataset(set_train_1.data,set_train_1.labels);
valset=prdataset(set_val_1.data,set_val_1.labels);
testset=prdataset(set_test_1.data,set_test_1.labels);

scatterdui(trainset, 'legend','.');
axis equal
print -r600 -dpng scattertrain.png % create a png file with the diagram
scatterdui(valset, 'legend','.');
print -r600 -dpng scatterval.png % create a png file with the diagram

figure;
scatterdui(testset, 'legend','.');
print -r600 -dpng scattertest.png % create a png file with the diagram

%%
figure;
varst = pcam(trainset,0);
stem(varst);
print -r600 -dpng 1_2.png % create a png file with the diagram

%% 
%creating trainset
wpca =pcam(trainset,2);
pca2_trainset=trainset*wpca;
scatterd(pca2_trainset,'.','legend');
print -r600 -dpng 1_3.png % create a png file with the diagram

%%
close all;
%figure;
for D=1:28
    wpcal =pcam(trainset,D);
     wpcalv =pcam(valset,D);
    wldcl=ldc(trainset*wpcal);
    Error_ratel(D)=trainset*wpcal*wldcl*testc;
    Error_ratelv(D)=valset*wpcalv*wldcl*testc;
      wqdcq=qdc(trainset*wpcal);
    Error_rateq(D)=trainset*wpcal*wqdcq*testc;
    Error_rateqv(D)=valset*wpcalv*wqdcq*testc;
    
end
[minvall,indexl]=min(Error_ratel);
[minvallv,indexlv]=min(Error_ratelv);

[minvalq,indexq]=min(Error_rateq);
[minvalqv,indexqv]=min(Error_rateqv);

Dl_opt=indexl;
Dq_opt=indexq;
Dl_optv=indexlv;
Dq_optv=indexqv;


d_plot=1:28;
plot(d_plot,Error_ratel,d_plot,Error_ratelv);
title('Error rate of ldc - trainset and valset');

legend('trainset',' valset');
print -r600 -dpng 2_1a.png 

figure;


plot(d_plot,Error_rateq,d_plot,Error_rateqv);
title('Error rate of qdc - trainset and valset');

legend('trainset',' valset');
print -r600 -dpng 2_1b.png 

%% 
wpca_final=pcam(trainset,3);
w_final=wpca_final*qdc(trainset*wpca_final);
C1=confmat(testset*w_final);

load image_features1.mat
im=im2feat(image_features);
imclass=im*w_final*classim;% applying training on image with color map
figure
show(imclass);
print -r600 -dpng classified_image.png


cost=[-1 0 0 0;0 -1 0 0;0 0 -1 0;0 0 0 -1];
testset_withcost = testset*w_final*classc*costm([],cost);
confmat(testset_withcost);
im_classified1 = im*w_final*classc*costm([],cost)*classim;
figure
show(im_classified1);
print -r600 -dpng classified_image1.png


%% 

net=patternnet(10);%one hiddenlayer with 10 neurons
pool.data = [set_train_2.data; set_val_2.data]';
pool.labels = [set_train_2.labels; set_val_2.labels]';
pool.labels = ind2vec(pool.labels);
testset2.data = set_test_2.data';
testset2.labels = set_test_2.labels';
testset2.labels = ind2vec(testset2.labels);

net=train(net,pool.data,pool.labels);
%% 
poolpred = net(pool.data);
pool_errorrate = perform(net,pool.labels,poolpred);
testpred = net(testset2.data);
test_errorrate = perform(net,testset2.labels,testpred);

%% 
load image_features2.mat
imdata1 = reshape(image_features_1,[256*256,28])';
impred1 = net(imdata1);
imclass1 = reshape(vec2ind(impred1),[256 256]);
figure
imagesc(imclass1); colormap(hot)
print -r600 -dpng imclass1.png

imdata2 = reshape(image_features_2,[256*256,28])';
impred2 = net(imdata2);
imclass2 = reshape(vec2ind(impred2),[256 256]);
figure
imagesc(imclass2); colormap(hot)
print -r600 -dpng imclass2.png




