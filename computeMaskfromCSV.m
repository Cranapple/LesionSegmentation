function computeMaskfromCSV()

%rootCSV = '../Data/adc/';
%destDir = '../Data/adc/masks/';

rootCSV = '../Data/t2/';
destDir = '../Data/t2/masks/';


list = dir(strcat(rootCSV, '*.csv'));
for i=1:length(list)
    fnameCsvAdc = strcat(rootCSV, list(i).name);
    
    listDicom = subdir(strcat(rootCSV, num2str(i), '/*.dcm'));
    info = dicominfo(listDicom(1).name);
    sizeX = double(info.Height);
    sizeY = double(info.Width);
    
    [tmp, ~] = readtext(fnameCsvAdc);
    
    slices = cell2mat(tmp(2:end,1))+1;
    
    for j=1:numel(slices)
        tmpPts = [];
        xPts{j} = cell2mat(tmp(j+1,19:5:end));
        yPts{j} = cell2mat(tmp(j+1,20:5:end));
    end
    
    uslices = unique(slices);
    
    for j=1:length(uslices)
        mask{j} = zeros(sizeX,sizeY);
        
        idxZ = find(uslices(j) == slices);
        for k=1:numel(idxZ)
            slice = slices(idxZ(k));
            mask{j} = mask{j} + poly2mask(round(xPts{idxZ(k)})+1,round(yPts{idxZ(k)})+1, sizeX, sizeY);
        end
    end
        
    fnameMatMask = sprintf('%s%02d.mat', destDir, i);
    save(fnameMatMask, 'mask', 'uslices');

    clear mask uslices xPts yPts slices tmpPts tmp;
end