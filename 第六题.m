% MNIST Digit Classification
% SGD vs Diagonal Natural Gradient (Fisher Approximation)

clear; clc; close all;
rng(1);

%% Load Dataset (MATLAB DigitDataset)

digitDatasetPath = fullfile(matlabroot,...
'toolbox','nnet','nndemos',...
'nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath,...
'IncludeSubfolders',true,...
'LabelSource','foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds,0.8,'randomized');

%% Convert to Matrix

numTrain = numel(imdsTrain.Files);
numTest  = numel(imdsTest.Files);

Xtrain = zeros(784,numTrain);
Ytrain = zeros(1,numTrain);

for k = 1:numTrain
    img = readimage(imdsTrain,k);
    img = imresize(img,[28 28]);
    Xtrain(:,k) = double(img(:))/255;
    Ytrain(k) = str2double(string(imdsTrain.Labels(k)));
end

Xtest = zeros(784,numTest);
Ytest = zeros(1,numTest);

for k = 1:numTest
    img = readimage(imdsTest,k);
    img = imresize(img,[28 28]);
    Xtest(:,k) = double(img(:))/255;
    Ytest(k) = str2double(string(imdsTest.Labels(k)));
end

Ntrain = numTrain;
Ntest  = numTest;

%% One-hot encoding

Ttrain = zeros(10,Ntrain);
for i=1:Ntrain
    Ttrain(Ytrain(i)+1,i)=1;
end

Ttest = zeros(10,Ntest);
for i=1:Ntest
    Ttest(Ytest(i)+1,i)=1;
end

%% Network

input_dim = 784;
hidden_dim = 200;
output_dim = 10;

%% SGD parameters

W1_sgd = 0.01*randn(hidden_dim,input_dim);
b1_sgd = zeros(hidden_dim,1);
W2_sgd = 0.01*randn(output_dim,hidden_dim);
b2_sgd = zeros(output_dim,1);

%% DNG parameters (Diagonal Fisher)

W1_dng = W1_sgd;
b1_dng = b1_sgd;
W2_dng = W2_sgd;
b2_dng = b2_sgd;

F1 = zeros(size(W1_dng));
F2 = zeros(size(W2_dng));
Fb1 = zeros(size(b1_dng));
Fb2 = zeros(size(b2_dng));

beta = 0.99;
eps0 = 1e-8;

%% Hyperparameters

Epoch = 50;
BatchSize = 128;

lr_sgd = 0.05;
lr_dng = 0.005;

%% Record

LossSGD = zeros(Epoch,1);
AccSGD  = zeros(Epoch,1);
LossDNG = zeros(Epoch,1);
AccDNG  = zeros(Epoch,1);

%% SGD Training

disp('Training SGD...')

for ep=1:Epoch

lr = lr_sgd*(0.98^(ep-1));
idx = randperm(Ntrain);

for k=1:BatchSize:Ntrain

id = idx(k:min(k+BatchSize-1,Ntrain));
X = Xtrain(:,id);
T = Ttrain(:,id);
B = size(X,2);

Z1 = W1_sgd*X + b1_sgd;
H = max(Z1,0.01*Z1);
Z2 = W2_sgd*H + b2_sgd;

Z2 = Z2 - max(Z2,[],1);
P = exp(Z2); P = P./sum(P,1);

dZ2 = (P-T)/B;
dW2 = dZ2*H';
db2 = sum(dZ2,2);

dH = W2_sgd'*dZ2;
dZ1 = dH;
dZ1(Z1<0)=0.01*dZ1(Z1<0);

dW1 = dZ1*X';
db1 = sum(dZ1,2);

W2_sgd = W2_sgd - lr*dW2;
b2_sgd = b2_sgd - lr*db2;
W1_sgd = W1_sgd - lr*dW1;
b1_sgd = b1_sgd - lr*db1;

end

[LossSGD(ep),AccSGD(ep)] = evaluateNet(Xtest,Ttest,W1_sgd,b1_sgd,W2_sgd,b2_sgd);

fprintf('SGD Epoch %d Acc %.4f\n',ep,AccSGD(ep));

end

%% DNG Training (Diagonal Fisher)

disp('Training DNG...')

for ep=1:Epoch

lr = lr_dng*(0.98^(ep-1));
idx = randperm(Ntrain);

for k=1:BatchSize:Ntrain

id = idx(k:min(k+BatchSize-1,Ntrain));
X = Xtrain(:,id);
T = Ttrain(:,id);
B = size(X,2);

Z1 = W1_dng*X + b1_dng;
H = max(Z1,0.01*Z1);
Z2 = W2_dng*H + b2_dng;

Z2 = Z2 - max(Z2,[],1);
P = exp(Z2); P = P./sum(P,1);

dZ2 = (P-T)/B;
dW2 = dZ2*H';
db2 = sum(dZ2,2);

dH = W2_dng'*dZ2;
dZ1 = dH;
dZ1(Z1<0)=0.01*dZ1(Z1<0);

dW1 = dZ1*X';
db1 = sum(dZ1,2);

% Fisher diagonal update
F1 = beta*F1 + (1-beta)*(dW1.^2);
F2 = beta*F2 + (1-beta)*(dW2.^2);
Fb1 = beta*Fb1 + (1-beta)*(db1.^2);
Fb2 = beta*Fb2 + (1-beta)*(db2.^2);

W1_dng = W1_dng - lr*dW1./sqrt(F1+eps0);
W2_dng = W2_dng - lr*dW2./sqrt(F2+eps0);
b1_dng = b1_dng - lr*db1./sqrt(Fb1+eps0);
b2_dng = b2_dng - lr*db2./sqrt(Fb2+eps0);

end

[LossDNG(ep),AccDNG(ep)] = evaluateNet(Xtest,Ttest,W1_dng,b1_dng,W2_dng,b2_dng);

fprintf('DNG Epoch %d Acc %.4f\n',ep,AccDNG(ep));

end

%% Evaluation function

function [loss,acc] = evaluateNet(X,T,W1,b1,W2,b2)

Z1 = W1*X + b1;
H = max(Z1,0.01*Z1);
Z2 = W2*H + b2;

Z2 = Z2 - max(Z2,[],1);
P = exp(Z2);
P = P./sum(P,1);

loss = -sum(sum(T.*log(P+1e-12)))/size(X,2);

[~,p] = max(P,[],1);
[~,t] = max(T,[],1);

acc = mean(p==t);

end