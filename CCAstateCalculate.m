function state = CCAstateCalculate(neighborState, continousConstant)
    
    %state = mean(neighborState) + continousConstant; 
    temp = mean(mean(neighborState)) + 0.45;
    state = temp - floor(temp) + continousConstant;

end