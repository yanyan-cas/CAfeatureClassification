function output = dummyClassifier(testSet, trainLabel, method)
    switch method
        case 'zeroRule'
            temp1 = categorical(string(trainLabel));
            [~,Categories] = histcounts(temp1);
            %temp2 = find(N==max(N));
            dummyPreLabel = Categories(1);                     
             for i = 1 : size(testSet, 1)
                output(i,1) = dummyPreLabel;
             end
            
        case 'Random'
            tempx = ['N', 'P'];
            %dummyPreLabel = tempx(unidrnd(2));
           	%dummyPreLabel = cellstr(trainLabel(size(trainLabel,1)));
            for i = 1 : size(testSet, 1)
                output(i,1) = tempx(unidrnd(2));
            end
        otherwise
            disp('errpr');
    end
           
   
    
end
