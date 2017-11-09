
clear all
clc
% %%
% % Digit Data Set: CIFAR
% 
 fprintf('Loading CIFAR data file\n'); 
 addpath('/Users/Yanyan/Documents/MATLAB/DATASET/cifar-10-batches-mat');
% % 
% % 
 raw = load('data_batch_1.mat');
 trainRaw = raw.data;
 trainLabel = raw.labels;
% 
% Raw2 = load('test_batch.mat');
% testRaw = Raw2.data;
% testLabel = Raw2.labels;
% 
% 
% 
% 
% 
% %%
% % Image RGB2GREY processing
% The first 1024 entries contain the red channel values, the next 1024 the green,
% and the final 1024 the blue. 
% The image is stored in row-major order, so that the first 32 entries of the array 
% are the red channel values of the first row of the image.

% tmp = zeros(1,32*32,3);
% trainData = zeros(size(trainRaw,1), 32 *32);
% for i = 1 : size(trainRaw,1)
%     tmp = reshape(trainRaw(i,:), [1,32*32,3]);
%     tmp2 = rgb2gray(tmp);
%     tmp3 = imbinarize(tmp2);
%     trainData(i,:) = double(tmp3);
% end

% testData = zeros(size(testRaw,1), 32 *32);
% for i = 1 : size(testData,1)
%     tmp = reshape(testRaw(1,:), [1,32*32,3]);
%     tmp2 = rgb2gray(tmp);
%     tmp3 = imbinarize(tmp2);
%     testData(i,:) = double(tmp3);
% end
% 
% %%
% % Resize the trainData
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
% %%
% % Calculate the evolution states
% CAEvol = 15;
% ruleNo = 127;
% extendCATrain = zeros(size(extendTrain, 1), 128*128);
% % 
% % for i = 1 : size(extendTrain, 1)
% %     extendCAtemp = extendTrain(i,:);
% % for j = 1 : CAEvol
% %         if j == 1
% %             output = CAtwoDimensionFeature(reshape(extendCAtemp, 128, 128)', ruleNo);
% %         else
% %             output = CAtwoDimensionFeature(output, ruleNo);
% %         end
% % end
% % extendCATrain(i,:) = reshape(output', 1, 128*128);
% % end
% 
% 
% for j = 1 : CAEvol
%     for i = 1 : size(extendTrain, 1)
%        if j ==1
%            output = CAtwoDimensionFeature(reshape(extendTrain(i,:), 128, 128)', ruleNo);
%        else
%            output =  CAtwoDimensionFeature(reshape(extendCATrain(i,:), 128, 128)', ruleNo);
%        end
%        extendCATrain(i,:) = reshape(output', 1, 128*128);
%     end
%     filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, j);
%     save(filename, 'extendCATrain');
% end


%%


CAEvol =15;

%for evolve =8 : CAEvol
ruleNo = 511;
evolve = 1;

    filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve);
    TEMP = load(filename, 'extendCATrain');
 
%%
% Baseline Evalutation <with data gray scaled and binaried>
% 

REPEAT = 1;

[~, labelItera] = histcounts(categorical(trainLabel));
cvErr = zeros(1, size(labelItera,2));
averageErr = zeros(REPEAT, 10);

X = double(TEMP.extendCATrain);
Y = cellstr(num2str(trainLabel)); 
group = char(Y);

for rep = 1 : REPEAT
    j = 0;
    
   %  c = cvpartition(group,'KFold', 10);    
     %acc = zeros(1, c.NumTestSets);
     KNNMdl = fitcknn(X,Y, 'Standardize',1 , 'CrossVal', 'on', 'Distance', 'hamming');
       % Mdl = fitcecoc(X, Y,  'CrossVal', 'on','CVPartition', c, 'Verbose', 2, 'Learner', 'knn', 'Coding', 'onevsall');
        averageErr(rep, :) = kfoldLoss(KNNMdl, 'Mode','individual');

end
filestorename = sprintf( 'exptest-kNNwithBinary%d_%dth_CIFAR.mat', ruleNo, evolve);
save(filestorename, 'averageErr');

%end


% %%
% % Experiments
% REPEAT = 10;
% 
% %Mdl = fitcknn(trainData, trainLabel);
% Mdl =  fitcecoc (double(trainData), trainLabel);
% ytest = predict(Mdl, double(testData));
% 
% EVAL = evaluate(char(testLabel), char(ytest));
% fprintf('accuracy is %f', EVAL(1));














