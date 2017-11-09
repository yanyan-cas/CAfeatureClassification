
clear all
clc
%%
% Digit Data Set: CIFAR

fprintf('Loading CIFAR data file\n'); 
addpath('/Users/Yanyan/Documents/MATLAB/DATASET/cifar-10-batches-mat');


raw = load('data_batch_1.mat');
trainRaw = raw.data;
trainLabel = raw.labels;



X = double(trainRaw);
Y = cellstr(num2str(trainLabel)); 
%group = char(Y);



 KNNMdl = fitcknn(X,Y, 'Standardize',1 , 'CrossVal', 'on');
       % Mdl = fitcecoc(X, Y,  'CrossVal', 'on','CVPartition', c, 'Verbose', 2, 'Learner', 'knn', 'Coding', 'onevsall');
        averageErr= kfoldLoss(KNNMdl, 'Mode','individual');
        
        filestorename = sprintf( 'exptest-kNNwithRawEuclidean.mat');
save(filestorename, 'averageErr');
