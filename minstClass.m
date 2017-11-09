clear all
clc
%%
% Digit Data Set: MNIST-Train

fprintf('Loading MNIST data file\n'); 
addpath('/Users/Yanyan/Documents/MATLAB/DATASET/digits');


rawZero = load('digit0.mat');
rawOne = load('digit1.mat');
rawTwo = load('digit2.mat');
rawThree= load('digit3.mat');
rawFour = load('digit4.mat');
rawFive= load('digit5.mat');
rawSix = load('digit6.mat');
rawSeven = load('digit7.mat');
rawEight = load('digit8.mat');
rawNine =  load('digit9.mat');

TRSIZE = 1000;
trainData = vertcat(rawZero.D(1:TRSIZE,:), rawOne.D(1:TRSIZE,:), rawTwo.D(1:TRSIZE,:), rawThree.D(1:TRSIZE,:), ...
             rawFour.D(1:TRSIZE,:), rawFive.D(1:TRSIZE,:), rawSix.D(1:TRSIZE,:), rawSeven.D(1:TRSIZE,:), rawEight.D(1:TRSIZE,:), rawNine.D(1:TRSIZE,:));
        
trainLabel = vertcat(zeros(TRSIZE,1), ones(TRSIZE,1), 2 * ones(TRSIZE,1), 3 * ones(TRSIZE,1), 4 * ones(TRSIZE,1), 5 * ones(TRSIZE,1), 6 *ones(TRSIZE,1),...
             7 * ones(TRSIZE,1), 8 * ones(TRSIZE,1), 9 * ones(TRSIZE,1));

clear rawZero rawOne rawTwo rawThree rawFour rawFive rawSix rawSeven rawEight rawNine

%%
% MNIST-Test
testrawZero = load('test0.mat');
testrawOne = load('test1.mat');
testrawTwo = load('test2.mat');
testrawThree= load('test3.mat');
testrawFour = load('test4.mat');
testrawFive= load('test5.mat');
testrawSix = load('test6.mat');
testrawSeven = load('test7.mat');
testrawEight = load('test8.mat');
testrawNine =  load('test9.mat');
TESIZE = 400;

testData = vertcat(testrawZero.D(1:TESIZE,:), testrawOne.D(1:TESIZE,:), testrawTwo.D(1:TESIZE,:), testrawThree.D(1:TESIZE,:), ...
             testrawFour.D(1:TESIZE,:), testrawFive.D(1:TESIZE,:), testrawSix.D(1:TESIZE,:), testrawSeven.D(1:TESIZE,:), testrawEight.D(1:TESIZE,:), testrawNine.D(1:TESIZE,:));
  
testLabel = vertcat(zeros(TESIZE,1), ones(TESIZE,1), 2 * ones(TESIZE,1), 3 * ones(TESIZE,1), 4 * ones(TESIZE,1), 5 * ones(TESIZE,1), 6 *ones(TESIZE,1),...
             7 * ones(TESIZE,1), 8 * ones(TESIZE,1), 9 * ones(TESIZE,1));

clear testrawZero testrawOne testrawTwo testrawThree testrawFour testrawFive testrawSix testrawSeven testrawEight testrawNine



%%
% adaptive
%trainData = imbinarize(trainRaw, 'adaptive');



REPEAT = 1;

averageErr = zeros(REPEAT, 10);

X = trainData;
Y = cellstr(num2str(trainLabel)); 
group = char(Y);

for rep = 1 : REPEAT
    j = 0;
    
     c = cvpartition(group,'KFold', 10);    
     %acc = zeros(1, c.NumTestSets);
     
        Mdl = fitcecoc(X, Y,  'CrossVal', 'on','CVPartition', c, 'Verbose', 2, 'Learner', 'knn');
        averageErr(rep, :) = kfoldLoss(Mdl, 'Mode','individual');

end

save('exp3-kNNwithMINST.mat', 'averageErr');


%%
% Baseline Classifier

REPEAT = 10;
%Mdl = fitcknn(trainData, trainLabel);
%Mdl =  fitcecoc (double(trainData), trainLabel);
%ytest = predict(Mdl, double(testData));

c = cvpartition(group,'KFold', 10);    
     acc = zeros(1, c.NumTestSets);
     
Mdl = fitcecoc(X, Y, 'CVPartition', c);
        kfoldLoss(Mdl, 'Mode','individual');
        
EVAL = evaluate(char(testLabel), char(ytest));
fprintf('baseline accuracy is %f\n', EVAL(1));

%%
% 





