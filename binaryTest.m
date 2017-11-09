
clear all
clc
%%
% Digit Data Set: CIFAR

fprintf('Loading CIFAR data file\n'); 
addpath('/Users/Yanyan/Documents/MATLAB/DATASET/cifar-10-batches-mat');


raw = load('data_batch_1.mat');
trainRaw = raw.data;
trainLabel = raw.labels;


%%





tmp = zeros(1,32*32,3);
trainData = zeros(size(trainRaw,1), 32 *32);
for i = 1 : size(trainRaw,1)
    tmp = reshape(trainRaw(i,:), [1,32*32,3]);
    tmp2 = rgb2gray(tmp);
    tmp3 = imbinarize(tmp2);
    trainData(i,:) = double(tmp3);
end

%%

% CAFeatureSize = 128 * 128;
% 
% CAtemp1 =  zeros(32, 48);
% CAtemp2 = zeros(48, 128);
% extendTrain = zeros(size(trainData, 1), CAFeatureSize);
% for inCA = 1 : size(trainData,1)
%     img = reshape(trainData(inCA,:),  32, 32)';
%     CAtemp = vertcat(CAtemp2, horzcat(CAtemp1, img, CAtemp1), CAtemp2);
%     %output = zeros(size(CAtemp,1), size(CAtemp,2));
%     extendTrain(inCA,:) = reshape(CAtemp', 1, CAFeatureSize);    
% end
% 
% output = zeros(size(CAtemp,1), size(CAtemp,2));
% 
% 


[~, labelItera] = histcounts(categorical(trainLabel));
cvErr = zeros(1, size(labelItera,2));


X = double(trainData);
Y = cellstr(num2str(trainLabel)); 
%group = char(Y);



 KNNMdl = fitcknn(X,Y , 'CrossVal', 'on');%'Distance', 'hamming'
       % Mdl = fitcecoc(X, Y,  'CrossVal', 'on','CVPartition', c, 'Verbose', 2, 'Learner', 'knn', 'Coding', 'onevsall');
        averageErr = kfoldLoss(KNNMdl, 'Mode','individual');
        
        filestorename = sprintf( 'exptest-kNNwithBinaryTestEuclideanNoStandard.mat');
save(filestorename, 'averageErr');
