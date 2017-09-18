function output = dummyClassifier(testSet, trainLabel, method)
    switch method
        case 'zeroRule'
            temp1 = categorical(string(trainLabel));
            [N,Categories] = histcounts(temp1);
            temp2 = find(N==max(N));
            dummyPreLabel = Categories(1);            
        case 'Random'
           	dummyPreLabel = cellstr(trainLabel(size(trainLabel,1)));
        otherwise
            disp('errpr');
    end
           
    for i = 1 : size(testSet, 1)
        output(i,1) = dummyPreLabel;
    end
    
end
