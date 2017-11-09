

clear all
clc
%%
% Digit Data Set: CIFAR

fprintf('Loading CIFAR data file\n'); 
addpath('/Users/Yanyan/Documents/MATLAB/DATASET/cifar-10-batches-mat');


raw = load('data_batch_1.mat');
trainRaw = raw.data;
trainLabel = raw.labels;

ruleNo = 31;
CAEvol = 15;

for evolve = 1 : CAEvol - 4


    filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve);
    TEMP1 = load(filename, 'extendCATrain');
    
    filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve+1);
    TEMP2 = load(filename, 'extendCATrain');
    
    filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve+2);
    TEMP3 = load(filename, 'extendCATrain');
    
    filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve+3);
    TEMP4 = load(filename, 'extendCATrain');
    
   % filename = sprintf('DataRuleNo%dEvolution%d', ruleNo, evolve+4);
    %TEMP5 = load(filename, 'extendCATrain');
    
    TEMP = [TEMP1.extendCATrain TEMP2.extendCATrain TEMP3.extendCATrain TEMP4.extendCATrain];
    clear TEMP1 TEMP2 TEMP3 TEMP4  trainRaw;
    
%%
% Baseline Evalutation <with data gray scaled and binaried>
% 

REPEAT = 1;

[~, labelItera] = histcounts(categorical(trainLabel));
cvErr = zeros(1, size(labelItera,2));
averageErr = zeros(REPEAT, 10);

X = double(TEMP);
Y = cellstr(num2str(trainLabel)); 
%group = char(Y);

for rep = 1 : REPEAT
    j = 0;
    
   %  c = cvpartition(group,'KFold', 10);    
     %acc = zeros(1, c.NumTestSets);
     KNNMdl = fitcknn(X,Y, 'Standardize',1 , 'CrossVal', 'on','Distance', 'hamming');
       % Mdl = fitcecoc(X, Y,  'CrossVal', 'on','CVPartition', c, 'Verbose', 2, 'Learner', 'knn', 'Coding', 'onevsall');
        averageErr(rep, :) = kfoldLoss(KNNMdl, 'Mode','individual');

end
filestorename = sprintf( 'exptest-kNNwithBinary%d_%dto%dth_CIFAR.mat', ruleNo, evolve, evolve+4);
save(filestorename, 'averageErr');

end