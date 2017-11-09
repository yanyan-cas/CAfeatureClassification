clc
clear all

%BASE = load('exptest-kNNwithRaw.mat');
%baseRaw = 1 - mean(BASE.averageErr);

BASE = load( 'exptest-kNNwithRawEuclidean.mat');
baseRawEuclidean = 1 - mean(BASE.averageErr);



BASE = load('exptest-kNNwithGrayTest.mat');
baseGray = 1 - mean(BASE.averageErr);

BASE = load('exptest-kNNwithBinaryTestEuclideanNoStandard.mat');
baseBinary = 1 - mean(BASE.averageErr);

%BASE = load('exptest-kNNwithExtendBinaryTest.mat')
%baseExtendBinary = 1 - mean(BASE.averageErr);

RuleNO = 511;
iteration = 15;

evolution = zeros(iteration, 1);
for i = 1 : iteration
    filename = sprintf( 'exptest-kNNwithBinary%d_%dth_CIFAR.mat', RuleNO, i);
  %  filename = sprintf( 'exptest-kNNwithBinaryFredkin_%dth_CIFAR.mat', i);
    ITERA = load(filename);
    evolution(i) = 1 - mean(ITERA.averageErr);
end

y = [baseRawEuclidean baseGray baseBinary  evolution'];
barGraph = bar(y, 0.5, 'c');

% saveas(gcf,'myfig.jpg');


grid on;
hold on
%set(gca,'XGrid','off');

set(gca, 'XTick', 1:18);
x = [1:1:18]';
for i1=1:numel(y)
    text(x(i1),y(i1),num2str(y(i1),'%0.2f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end

set(gca,'XTickLabel',{'RAW','GRAY','BINA','1st','2nd','3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th', '13th', '14th', '15th'});
xlabel('Experiments');% x轴名称
ylabel('Accuracy');
tilename = sprintf('Expereiments for 2-Dimensional Linear Rule %d', RuleNO);
title(tilename);


