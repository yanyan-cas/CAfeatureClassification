function output = extractCAFeatures(input, ruleNo, boundary)

    [m, n] = size(input);
   
    x = dec2bin(ruleNo, 9);
    % auxiliary matrices T_1 and T_2
    switch boundary
        case 'NullBoundary'
            T1 = horzcat(zeros(m, 1), vertcat(eye(m-1), zeros(1,m-1)));
            T2 = vertcat(zeros(1, n), horzcat( eye(n-1),  zeros(n-1, 1)));
            %base matrix
            base1NB = input;
            base2NB = input * T2;
            base4NB = T1 * input * T2;
            base8NB = T1 * input;
            base16NB = T1 * input * T2;
            base32NB = input * T1;
            base64NB = T2 * input *T1;
            base128NB = T2 * input;
            base256NB = T2 * input * T2;     
            
            temp = base1NB .* bin2dec(x(9)) + base2NB .* bin2dec(x(8)) + base4NB .* bin2dec(x(7)) + base8NB .* bin2dec(x(6)) + base16NB  .* bin2dec(x(5)) ... 
                     + base32NB .* bin2dec(x(4)) + base64NB  .* bin2dec(x(3)) + base128NB  .* bin2dec(x(2)) + base256NB  .* bin2dec(x(1));
                     
            next_state = mod(temp, 2);
            
        case 'PeriodicBoundary'
            T1 = horzcat(vertcat(zeros(m-1, 1), 1), vertcat(eye(m-1), zeros(1,m-1)));           
            T2 = vertcat(horzcat(zeros(1, n-1), 1), horzcat( eye(n-1),  zeros(n-1, 1)));
             %base matrix
            base1PB = input;
            base2PB = input * T2;
            base4PB = T1 * input * T2;
            base8PB = T1 * input;
            base16PB = T1 * input * T1;
            base32PB = input * T1;
            base64PB = T2 * input *T1;
            base128PB = T2 * input;
            base256PB = T2 * input * T2;
            
                
            temp = base1PB .* bin2dec(x(9)) + base2PB .* bin2dec(x(8)) + base4PB .* bin2dec(x(7)) + base8PB  .* bin2dec(x(6)) + ...
                    base16PB .* bin2dec(x(5)) + base32PB .* bin2dec(x(4)) + base64PB  .* bin2dec(x(3)) + base128PB  .* bin2dec(x(2)) + base256PB  .* bin2dec(x(1));
    
            next_state = mod(temp, 2);
            
        case 'AdiabaticBoundary'
            T1 =  horzcat(zeros(m, 1), vertcat(eye(m-1), horzcat(zeros(1,m-2),1)));            
            T2 = vertcat(horzcat(1, zeros(1, n-1)), horzcat(eye(n-1),  zeros(n-1, 1)));
             %base matrix
            base1AB = input;
            base2AB = input * T1';
            base4AB = T1 * input * T1';
            base8AB = T1 * input;
            base16AB = T1 * input * T2';
            base32AB = input * T2';
            base64AB = T2 * input *T2';
            base128AB = T2 * input;
            base256AB = T2 * input * T1';
            
            
            temp = base1AB .* bin2dec(x(9)) + base2AB .* bin2dec(x(8)) + base4AB .* bin2dec(x(7)) + base8AB  .* bin2dec(x(6)) + base16AB  .* bin2dec(x(5)) ... 
                 + base32AB .* bin2dec(x(4)) + base64AB  .* bin2dec(x(3)) + base128AB  .* bin2dec(x(2)) + base256AB  .* bin2dec(x(1));
    
            next_state = mod(temp, 2);
        case 'ReflexiveBoundary'
            T1 = horzcat(zeros(m, 1), vertcat(eye(m-1), horzcat(zeros(1,m-3),1,0)));          
            T2 = vertcat(horzcat(0, 1, zeros(1, n-2)), horzcat(eye(n-1),  zeros(n-1, 1)));
             %base matrix
            base1RB = input;
            base2RB = input * T1';
            base4RB = T1 * input * T1';
            base8RB = T1 * input;
            base16RB = T1 * input * T2';
            base32RB = input * T2';
            base64RB = T2 * input *T2';
            base128RB = T2 * input;
            base256RB = T2 * input * T1';
                               
            temp = base1RB .* bin2dec(x(9)) + base2RB .* bin2dec(x(8)) + base4RB .* bin2dec(x(7)) + base8RB  .* bin2dec(x(6)) + base16RB  .* bin2dec(x(5)) ... 
                  + base32RB .* bin2dec(x(4)) + base64RB  .* bin2dec(x(3)) + base128RB  .* bin2dec(x(2)) + base256RB  .* bin2dec(x(1));
            next_state = mod(temp, 2);
        otherwise
            
    end
    
    output = next_state;
    %output = reshape(next_state, [1, 64*64]);
    
  
end
