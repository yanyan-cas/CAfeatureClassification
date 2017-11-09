function state = stateCalculate(neighborState, ruleNumber)

    %% unpack rule
    %for ruleNo = 170, the bitget result is 0 1 0 1 0 1 0 1, but for
    %nine-neighbor settings: Rule 170NB = 2NB + 8NB + 32NB + 128NB which is
    % ( 0 1 0 1 0 1 0 1 )  <start from 2^0 to 2^7>
    
    %testcode: neighborState = eye(3);
    
    ruleArray = double(bitget(uint16(ruleNumber), 1:9)); 
    
    %% calculate next state
%    if neighborSettings == 9
        
        % The next state of the cell depends on the rulearr and state of
        % current neighbors, which I make a matrix multiplication for
        % conveince: x^(t+1) = ( ruleArray * neighbor state^t ) mod 2
       tmp = ruleArray(1) * neighborState(2,2) + ruleArray(2) * neighborState(2,3) + ruleArray(3) * neighborState(3,3) +...
           ruleArray(4) * neighborState(3,2) + ruleArray(5) * neighborState(3,1) + ruleArray(6) * neighborState(2,1) +...
           ruleArray(7) * neighborState(1,1) + ruleArray(8) * neighborState(1,2) + ruleArray(9) * neighborState(1,3);       
        state = mod(tmp, 2);           
%    else       
%        state = 0;
end