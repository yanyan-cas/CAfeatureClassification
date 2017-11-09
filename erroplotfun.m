%load baseline.mat;
%y = mean(cvErr);
%e = std(cvErr,1,2);


ruleNo = [128 257 324 394 402 417];
EVOLE = 16;
classCategory = 10;

for i = ruleNo
    for j = 1 : EVOLE
        filename = sprintf('rule%d_itera%d.mat', i, j);
        x = load(filename);
        for k = 1: classCategory
            temp(j, k) = x.averageErr;
        end
    end
    
end



load nonCALinear.mat
y = mean(averageErr, 1);
e = std(averageErr);
figure
errorbar(y,e,'or');

