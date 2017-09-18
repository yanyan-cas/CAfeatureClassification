%%
% Digit Data Set
% Load training and test data using |imageDatastore|.
% syntheticDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
% handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');
% 
% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.
% trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

fprintf('Loading semeion.data file\n'); 

raw = load('semeion.data.txt');
Data = raw(:,1:256);
rawLabel = raw(:,257:266);


Label = zeros(length(rawLabel(:,1)),1);
for i = 1 : length(rawLabel(:,1))
    Label(i) = find(rawLabel(i,:))-1;
end

X  = Data;
Y = cellstr(num2str(Label)); % We need a cell to used the ClassName parameter in fitcecoc

%%
% Baseline Evalutation

[~, labelItera] = histcounts(categorical(Label));
j = 0;
cvErr = zeros(1, size(labelItera,2));

for i = labelItera
    j = j + 1;
    x = char(i);
    expY = char(Y);
    expY(expY ~= x) = 'N';
    expY(expY == x) = 'P';

    group = expY;   
    c = cvpartition(group,'KFold', 10);    
    acc = zeros(1, c.NumTestSets);
for k = 1 : c.NumTestSets
        trainIndex = c.training(k);
        testIndex = c.test(k);
        ytest = dummyClassifier( X(testIndex, :), expY(trainIndex, :), 'zeroRule');      %zeroRule or Random for baseline assessment
        EVAL = evaluate(char(expY(testIndex)), char(ytest));
        % confusionmat(char(ytest), char(Y(testIndex)));
        %err(i) = sum(~strcmp(ytest, Y(testIndex)));
        acc(k) = EVAL(1);
end     
        cvErr(j) = sum(acc)/c.NumTestSets;
        fprintf('Digital number %s the baseline classfication accuracy is %f \n', x, cvErr(j) );

end



%%
% Linear Classifier Test with Original Input
% REPEAT = 10;
% 
% [a, labelItera] = histcounts(categorical(Label));
% expY = char(Y);
% cvErr = zeros(1, size(labelItera,2));
% averageErr = zeros(REPEAT, size(cvErr, 2));
% 
% for rep = 1 : REPEAT
%     j = 0;
% % fprintf('Repeat for the %d time:\n', rep);
% for i = labelItera
%     j = j + 1;
%     x = char(i);
%     expY = char(Y);
%     expY(expY ~= x) = 'N';
%     expY(expY == x) = 'P';
% 
%     group = expY;   
%     c = cvpartition(group,'KFold', 10);    
%     acc = zeros(1, c.NumTestSets);
% for k = 1 : c.NumTestSets
%         trainIndex = c.training(k);
%         testIndex = c.test(k);
%         Mdl = fitclinear(X(trainIndex, :), expY(trainIndex, :), 'Learner','logistic');
%         ytest = predict(Mdl, X(testIndex,:));
%         EVAL = evaluate(char(expY(testIndex)), char(ytest));
%         acc(k) = EVAL(1);
% end     
%         cvErr(j) = sum(acc)/c.NumTestSets;
%        % fprintf('Digital number %s the baseline classfication accuracy is %f \n', x, cvErr(j) );
% end
%         averageErr(rep,:) = cvErr;
% end

%%
% Experiment for Input Dimension Extended

CAFeatureSize = 64*64;

CAtemp1 =  zeros(16, 24);
CAtemp2 = zeros(24, 64);
extendX = zeros(size(X, 1), CAFeatureSize);
for inCA = 1 : size(X,1)
    img = reshape(X(inCA,:),  16, 16);
    CAtemp = vertcat(CAtemp2, horzcat(CAtemp1, img, CAtemp1), CAtemp2);
    output = zeros(size(CAtemp,1), size(CAtemp,2));
    extendX(inCA,:) = reshape(CAtemp, 1, CAFeatureSize);    
end
    
REPEAT = 10;

[a, labelItera] = histcounts(categorical(Label));
expY = char(Y);
cvErr = zeros(1, size(labelItera,2));
averageErr = zeros(REPEAT, size(cvErr, 2));

for rep = 1 : REPEAT
    j = 0;
% fprintf('Repeat for the %d time:\n', rep);
for i = labelItera
    j = j + 1;
    x = char(i);
    expY = char(Y);
    expY(expY ~= x) = 'N';
    expY(expY == x) = 'P';

    group = expY;   
    c = cvpartition(group,'KFold', 10);    
    acc = zeros(1, c.NumTestSets);
for k = 1 : c.NumTestSets
        trainIndex = c.training(k);
        testIndex = c.test(k);
        Mdl = fitclinear(extendX(trainIndex, :), expY(trainIndex, :), 'Learner','logistic');
        ytest = predict(Mdl, extendX(testIndex,:));
        EVAL = evaluate(char(expY(testIndex)), char(ytest));
        acc(k) = EVAL(1);
end     
        cvErr(j) = sum(acc)/c.NumTestSets;
       % fprintf('Digital number %s the baseline classfication accuracy is %f \n', x, cvErr(j) );
end
        averageErr(rep,:) = cvErr;
end






%     group = Y;   
%     c = cvpartition(group,'KFold', 10);    
%     
% for i = 1 : c.NumTestSets
%         trainIndex = c.training(i);
%         testIndex = c.test(i);
%         ytest = dummyClassifier( X(testIndex, :), Y(trainIndex, :));      
%         evaluate(char(Y(testIndex)), char(ytest));
%         confusionmat(char(ytest), char(Y(testIndex)));
%         %err(i) = sum(~strcmp(ytest, Y(testIndex)));
% end     





%cvErr = sum(err)/sum(c.TestSize);





%t = templateLinear('Lambda','auto', 'Learner','SVM','Regularization','lasso');
% a cross-validation  (e.g., CrossVal), then n is the number of in-fold
% observations -> auto

%c = cvpartition(group,'KFold',k); %group means equal size and distribution

%Mdl = fitcecoc(X, Y, 'Learners',t, 'CrossVal', 'on',  'KFold', 10,'Coding', 'onevsall', 'Verbose',1);
%kfoldLoss(Mdl, 'Mode', 'individual');

%%
%
    
ruleNo = 511;
neighbor = 'Moore';
boundary = 'NullBoundary';%PeriodicBoundary NullBoundary  AdiabaticBoundary  ReflexiveBoundary
%iteration = 1;
% CAtemp1 =  false(16, 24);
% CAtemp2 = false(24, 64);
% CAtemp = vertcat(CAtemp2, horzcat(CAtemp1, processedImage, CAtemp1), CAtemp2);
%imagesc(CAtemp);
%output = extractCAFeatures(CAtemp, ruleNo, boundary);
%imagesc(output);
%for i = 1 : 110
%    output = extractCAFeatures(output, ruleNo, boundary);
%    imagesc(output);
%end
%imagesc(output);
%figure;

%subplot(1,2,1)
%imshow(exTestImage)

%subplot(1,2,2)
%imshow(processedImage)

%%
% Using HOG Features
% img = readimage(trainingSet, 206);
% 
% % Extract HOG features and HOG visualization
% [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
% [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
% [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
% 
% % Show the original image
% figure;
% subplot(2,3,1:3); imshow(img);
% 
% % Visualize the HOG features
% subplot(2,3,4);
% plot(vis2x2);
% title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
% 
% subplot(2,3,5);
% plot(vis4x4);
% title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
% 
% subplot(2,3,6);
% plot(vis8x8);
% title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
% 
% cellSize = [4 4];
% hogFeatureSize = length(hog_4x4);

%%
% Train a Digit Classifier

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.



%imagesc(output);
time = 2;

CAFeatureSize = 64*64;
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, CAFeatureSize, time, 'single');
CAtemp1 =  false(16, 24);
CAtemp2 = false(24, 64);
trainDesignedFeatures = zeros(numImages, CAFeatureSize* time, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    img = rgb2gray(img);
    % Apply pre-processing steps
    img = imbinarize(img);
    CAtemp = vertcat(CAtemp2, horzcat(CAtemp1, img, CAtemp1), CAtemp2);
    output = false(size(CAtemp,1), size(CAtemp,2));
    
    for j = 1 : time
        if j == 1
            output = extractCAFeatures(CAtemp, ruleNo, boundary);
        else
            output = extractCAFeatures(output, ruleNo, boundary);
        end
        trainingFeatures(i, :, j) = reshape(output, [1, 64*64]);
        % operations to deal with the CA time iterations
        trainDesignedFeatures(i, CAFeatureSize*(j-1) + 1 : CAFeatureSize * j ) = trainingFeatures(i, :, j);
    end

    %trainingFeatures(i, :) = reshape(output, [1, 64*64]);
    %trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
% operations to deal with the CA time iterations
%trainDesignedFeatures = zeros(numImages, CAFeatureSize* time, 'single');

%for i = 1 : numImages
   % for j = 1 : time
        
      %  if j == 1
      %      trainDesignedFeatures(i, 1 : CAFeatureSize) = trainingFeatures(i, :, j);
      %  else           
         %   trainDesignedFeatures(i, CAFeatureSize*(j-1) + 1 : CAFeatureSize * j ) = trainingFeatures(i, :, j);
    %    end
        
   % end
%end
% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainDesignedFeatures, trainingLabels);

%%
% Evaluate the Digit Classifier
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
%[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);


numImagesTest = numel(testSet.Files);
testFeatures = zeros(numImagesTest, CAFeatureSize, time, 'single');
testDesignedFeatures = zeros(numImagesTest, CAFeatureSize* time, 'single');

for i = 1:numImagesTest
    img = readimage(testSet, i);

    img = rgb2gray(img);
    img = imbinarize(img);
    CATestTemp = vertcat(CAtemp2, horzcat(CAtemp1, img, CAtemp1), CAtemp2);
for j = 1 : time
         if j == 1
            output = extractCAFeatures(CATestTemp, ruleNo, boundary);
        else
            output = extractCAFeatures(output, ruleNo, boundary);
         end
end
    testFeatures(i, :, j) = reshape(output, [1, 64*64]);
    testDesignedFeatures(i, CAFeatureSize*(j-1) + 1 : CAFeatureSize * j ) = testFeatures(i, :, j);
end

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);
testLabels = testSet.Labels;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
size(find(testLabels == predictedLabels))
%helperDisplayConfusionMatrix(confMat);

confusionMatrixPlot;







