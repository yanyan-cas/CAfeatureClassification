load fisheriris
X = meas;
Y = species;

k=3;
KNNMdl = fitcknn(X,Y, 'NumNeighbors',k,'Standardize',1, 'CrossVal', 'on');


classError = loss(KNNMdl, 'mode', 'individual')
