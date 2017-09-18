function output = dummyClassifierMultiClass(testSet, trainLabel)
    temp1 = categorical(trainLabel);
    [N,Categories] = histcounts(temp1);
    temp2 = find(N==max(N));
    if size(temp2, 2) > 1
        temp3 = unidrnd(size(temp2, 2));
    else
        temp3 = temp2;
    end
    dummyPreLabel = Categories(temp3);
   % ouput = dummuPreLabel;
  % temp4 = zeros(size(testSet, 1), 1);
    %ouput = cell(size(temp4));
    for i = 1 : size(testSet, 1)
        output(i,1) = dummyPreLabel;
    end
    
end