function grayImage = pixelRGB2gray(pixelRGBrow)
    redChannel = pixelRGBrow(1:1024);
    greenChannel = pixelRGBrow(1025:2048);
    blueChannel = pixelRGBrow(2049:3072);
    
    grayImage = .299*double(redChannel) + ...
                  .587*double(greenChannel) + ...
                  .114*double(blueChannel);
              
    grayImage = uint8(grayImage);


end
