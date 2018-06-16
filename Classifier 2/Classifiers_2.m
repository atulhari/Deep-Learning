
close all % close all figure windows
clear variables % clear the workspace

load dataset_ex2
prwarning off % set the prtools warning level to 'off'
prwaitbar off % set the prtools waitbar level to 'off'
gridsize(200); % set the prtools gridsize for plot commands to ‘fine’
ctset_1 = prdataset(ctset1.data,ctset1.labels);
figure;
scatterd(ctset_1,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
print -r600 -dpng scattered.png

wldc = ldc(ctset_1); % create a Linear Bayes Normal Classifier
figure;
scatterd(ctset_1,'.'); % create a scatter diagram
legend('Soft tissue','iodine ','Bone','Location','SouthEast'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
plotm(wldc,2); % plot the contour lines of the gaussians
plotc(wldc,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng 1_2.png
error_rate_l = ctset_1*wldc*testc;%% Calculating error rate
wqdc = qdc(ctset_1); % create a Linear Bayes Normal Classifier
figure;
scatterd(ctset_1,'.'); % create a scatter diagram
legend('Soft tissue','iodine ','Bone','Location','SouthEast'); % create legend for scatterplot
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
plotm(wqdc,2); % plot the contour lines of the gaussians
plotc(wqdc,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng 1_2a.png
error_rate_q = ctset_1*wqdc*testc;%% Calculating error rate

%%
wldc_unt = ldc([]); % create an untrained ldc
wqdc_unt = qdc([]); % create an untrained qdc
[Eldc,Sldc] = prcrossval(ctset_1,wldc_unt,5,10); % 5-foldcross val with 10 repetitions
[Eqdc,Sqdc] = prcrossval(ctset_1,wqdc_unt,5,10); % 5-fold cross val with 10 repetitions


%% Q
close all

for k=1:10
    Eknnc_blunt(k) = ctset_1*knnc(ctset_1,k)*testc; % test and train with the same set
    wkkc_unt = knnc([],k); % create untrained knn classifier
    Eknnc_cv(k) = prcrossval(ctset_1,wkkc_unt,5,10); % apply cross validation
end
figure;
plot(1:10,Eknnc_blunt,1:10,Eknnc_cv);
xlabel('k'); ylabel('error rate');
legend('blunt','cross validation')
print -r600 -dpng 1_6.png

%%

%%
tset=ctset_1;
[sens,spec]=testc(tset,wldc,'sensitivity',2);
prec=testc(tset,wldc,'precision',2);
auc=testc(tset,wldc,'auc',2);
TP= testc(tset,wldc,'TP',2);
FN= testc(tset,wldc,'FN',2);
FP=(TP/prec)-TP;
error_rate=(FP+FN)/400;
accuracy=1-error_rate;

C=confmat(tset*wldc);


%%

ctset1a = prdataset(ctset1.data,ctset1.labels);
ctset2a = prdataset(ctset2.data,ctset2.labels);

figure;
scatterd(ctset1a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 1')
wldc1 = ldc(ctset1a); % create a Linear Bayes Normal Classifier
plotc(wldc1,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3a.png

figure;
scatterd(ctset2a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 2')
wldc2 = ldc(ctset2a); % create a Linear Bayes Normal Classifier
plotc(wldc2,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3b.png

%% Errors
error_ldc1=ctset1a*wldc*testc ;% test on set 1
wldc_unt = ldc([]); % create an untrained ldc
[E_ldc1,S_ldc1] = prcrossval(ctset1a,wldc_unt,5,10); % 5-foldcross val with 10 repetitions
error_ldc2=ctset2a*wldc*testc ;% test on set 2



figure;
scatterd(ctset1a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 1')
wqdc1 = qdc(ctset1a); % create a Linear Bayes Normal Classifier
plotc(wqdc1,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3aq.png

figure;
scatterd(ctset2a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 2')
wqdc2 = qdc(ctset2a); % create a Linear Bayes Normal Classifier
plotc(wqdc2,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3bq.png

%% Errors
error_qdc1=ctset1a*wqdc*testc; % test on set 1
wqdc_unt = qdc([]); % create an untrained ldc
[E_qdc1,S_qdc1] = prcrossval(ctset1a,wqdc_unt,5,10); % 5-foldcross val with 10 repetitions
error_qdc2=ctset2a*wqdc*testc ;% test on set 2

%%
[W1,k,E]=knnc(ctset1a);

figure;
scatterd(ctset1a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 1')
plotc(W1,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3ak.png

figure;
scatterd(ctset2a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 2')

plotc(W1,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scattered3bk.png



error_knnc1=ctset1a*W1*testc;
W1_unt=scalem([],'variance')*knnc([]);
[E_knnc1,S_knnc1] = prcrossval(ctset1a,W1_unt,5,10); % 5-foldcross val with 10 repetitions
error_knnc2=ctset2a*W1*testc; % test on set 2



%%
n=prtime(1);

svcpoly_unt = svc([],'p',10);
svcgauss_unt = svc([],'r',0.1);
[poly_cv,poly_sd] = prcrossval(ctset1a,svcpoly_unt,5,10); % apply cross validation
[gauss_cv,gauss_sd] = prcrossval(ctset1a,svcgauss_unt,5,10); % apply cross validation
d=10;
r=0.1;
w_svcpoly= svc(ctset1a,'p',d);
svcpoly_unt = svc([],'p',d);
w_svcgauss = svc(ctset1a,'r',r);
svcgauss_unt = svc([],'r',r);
figure;
scatterd(ctset1a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w_svcpoly); % plot the contour lines of the gaussians
title('Dataset 1')
print -r600 -dpng svcpolyset1.png
figure;
scatterd(ctset2a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w_svcpoly); % plot the contour lines of the gaussians
title('Dataset 2')
print -r600 -dpng svcpolyset2.png
error_rate_svcp1 = ctset1a*w_svcpoly*testc;
error_rate_svcp2 = ctset2a*w_svcpoly*testc;
[svcpoly_cv,svcpoly_sd] = prcrossval(ctset1a,svcpoly_unt,5,10); % apply cross validation


figure;
scatterd(ctset1a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w_svcgauss); % plot the contour lines of the gaussians
title('Dataset 1')
print -r600 -dpng svcgaussset1.png
figure;
scatterd(ctset2a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w_svcgauss); % plot the contour lines of the gaussians
title('Dataset 2')
print -r600 -dpng svcgaussset2.png
error_rate_svcg1 = ctset1a*w_svcgauss*testc;
error_rate_svcg2 = ctset2a*w_svcgauss*testc;
[svcgauss_cv,svcgauss_sd] = prcrossval(ctset1a,svcgauss_unt,5,10); % apply cross validation

%%
[w, history, units] = bpxnc(ctset1a);
plot(history(:,1),history(:,2),history(:,1),history(:,4))
legend('error rate','mse')
title('Error rate and mse vs epoch number')
xlabel('epoch number')
yyaxis left
ylabel('error rate')
yyaxis right
ylabel('mse')
print -r600 -dpng epoch.png
figure;
scatterd(ctset1a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w); % plot the contour lines of the gaussians
title('Dataset 1')
print -r600 -dpng nueralset1.png
figure;
scatterd(ctset2a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w); % plot the contour lines of the gaussians
title('Dataset 2')
print -r600 -dpng nueralset2.png
error_rate_nn1_1 = ctset1a*w*testc;
error_rate_nn1_2 = ctset2a*w*testc;
nn_untrained = bpxnc([],40);
[nn1_cv,nn1_sd] = prcrossval(ctset1a,nn_untrained,5,10); % apply cross validation
[w2, history2] = bpxnc(ctset1a, [40 20]);
plot(history2(:,1),history2(:,2),history2(:,1),history2(:,4))
legend('error rate','mse')
title('Error rate and mse vs epoch number')
xlabel('epoch number')
yyaxis left
ylabel('error rate')
yyaxis right
ylabel('mse')
print -r600 -dpng epochhidden2.png
figure;
scatterd(ctset1a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w2); % plot the contour lines of the gaussians
title('Dataset 1')
print -r600 -dpng nueralset1hidden2.png
figure;
scatterd(ctset2a,'.'); % create a scatter diagram
legend('Soft tissue', 'Iodine', 'Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
plotc(w2); % plot the contour lines of the gaussians
title('dataset 2')
print -r600 -dpng nueralset2hidden2.png
error_rate_nn_set1_double_layer = ctset1a*w2*testc;
error_rate_nn_set2_double_layer = ctset2a*w2*testc;

nn_untrained2 = bpxnc([],[40 20]);
[nn2_cv,nn2_sd] = prcrossval(ctset1a,nn_untrained2,5,10); % apply cross validation

%%


W_tree=treec(ctset1a,'infcrit',10);
W_tree_unt=treec([],'infcrit',10);
[E_tree1,S_tree1] = prcrossval(ctset1a,W_tree_unt,5,10) ;% 5-foldcross val with 10 repetitions

figure;
scatterd(ctset1a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 1')
plotc(W_tree,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scatteredtreea.png

figure;
scatterd(ctset2a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 2')

plotc(W_tree,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scatteredtreeb.png

error_tree1=ctset1a*W_tree*testc;
error_tree2=ctset2a*W_tree*testc; % test on set 2


%%

W_rf=randomforestc(ctset1a,183);
W_rf_unt=randomforestc([],183);
[E_rf1,S_rf1] = prcrossval(ctset1a,W_rf_unt,5,10) ;% 5-foldcross val with 10 repetitions

figure;
scatterd(ctset1a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 1')
plotc(W_rf,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scatteredrfa.png

figure;
scatterd(ctset2a,'legend','.');
legend('Soft tissue','iodine ','Bone','Location','SouthEast');
axis equal % make sure that axis scales are equal
set(gca,'fontsize',12); % set the font size to something reasonable
title('Dataset 2')

plotc(W_rf,'g'); % plot the decision boundaries of a classifier
print -r600 -dpng scatteredrfb.png

error_rf1=ctset1a*W_rf*testc;
error_rf2=ctset2a*W_rf*testc; % test on set 2

%%

ctscandata=im2feat(ctim);%% converting array to data
figure;
show(ctscandata);
print -r600 -dpng ctscandat.png
figure;
show(ctscandata*wldc);%% using ldc
print -r600 -dpng ctscandat_ldc.png
figure;
classim(ctscandata*wldc);%% classification using classim
% show(imclass);
print -r600 -dpng ctscandat_classimldc.png
figure;
show(ctscandata*W_rf);%% using randomforest
print -r600 -dpng ctscandat_rf.png
figure
classim(ctscandata*W_rf);%% classification using classim
print -r600 -dpng ctscandat_classim_rf.png


