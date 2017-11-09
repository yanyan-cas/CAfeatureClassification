function nextImageState = CAtwoDFeatureFredkin(inputMatrix)
    [m, n] = size(inputMatrix);
    seed = inputMatrix;   
%%
% calculate the next state for the Non-boundary cells -- central part
for i = 2 : m-1
    for j = 2 : n-1
        % get the neighbor cells' states
        temp = [ seed(i-1, j-1),  seed(i-1, j),   seed(i-1, j+1);...
                        seed(i, j-1),     seed(i, j), seed(i, j+1); ...
                        seed(i+1, j-1), seed(i+1, j) , seed(i+1, j+1)];
        nborState = temp;
%         temp = [1, 2, 3, 4 , 5, 6, 7, 8, 9];
%         nborState = reshape(temp', 3, 3);
%          % get the next state
        nextImageState(i, j) = FredkinStateCalculate(nborState);        
    end
end

%%
% calculate the next state for the Non-boundary cells -- first row
for i = 1
    for j = 1 : n
        % get the neighbor cells' states - for NullBoundary the neighbor
        % cells' states are 0
        switch j         
            case 1
                 temp = [  0,               0,               0; ...
                                 0,         seed(i, j),     seed(i, j+1); ...
                                 0,         seed(i+1, j) , seed(i+1, j+1) ];               
            case n
                 temp = [  0,                      0,              0; ...
                        seed(i, j-1),     seed(i, j),           0; ...
                        seed(i+1, j-1), seed(i+1, j) ,       0];                    
            otherwise   
                temp = [  0,                      0,              0; ...
                        seed(i, j-1),     seed(i, j), seed(i, j+1); ...
                        seed(i+1, j-1), seed(i+1, j) , seed(i+1, j+1) ];    
        end
        nborState = temp;
         % get the next state             
       nextImageState(i, j) = FredkinStateCalculate(nborState);   
    end    
end
    
%%
% calculate the next state for the Non-boundary cells -- LAST row
for i = m
    for j = 1 : n
      % get the neighbor cells' states - for NullBoundary the neighbor
        % cells' states are 0
        switch j         
            case 1
                 temp = [  0,        seed(i-1, j),   seed(i-1, j+1); ...
                               0,           seed(i, j),      seed(i, j+1);...
                               0,                  0 ,                       0 ];                 
            case n
                 temp = [ seed(i-1, j-1),  seed(i-1, j),   0;...
                        seed(i, j-1),     seed(i, j),              0;...
                             0,                  0 ,                       0 ];                   
            otherwise   
                temp = [  seed(i-1, j-1),  seed(i-1, j),   seed(i-1, j+1); ...
                        seed(i, j-1),     seed(i, j), seed(i, j+1);...
                               0,                0,               0];    
        end
        nborState = temp;
         % get the next state             
       nextImageState(i, j) = FredkinStateCalculate(nborState);   
       
    end
end
    
%%
% calculate the next state for the Non-boundary cells -- first column and
% last column

for i = 2 : m -1
    for j = [1 n]
          % get the neighbor cells' states
          if j == 1
        temp = [ 0,  seed(i-1, j),   seed(i-1, j+1);...
                        0,     seed(i, j), seed(i, j+1); ...
                        0, seed(i+1, j) , seed(i+1, j+1) ];    
          else 
                % get the neighbor cells' states
        temp = [ seed(i-1, j-1),  seed(i-1, j),   0; ...
                        seed(i, j-1),     seed(i, j), 0; ...
                        seed(i+1, j-1), seed(i+1, j) , 0 ];    
          end
            nborState = temp;
         % get the next state             
       nextImageState(i, j) = FredkinStateCalculate(nborState);   
    end
end

end
