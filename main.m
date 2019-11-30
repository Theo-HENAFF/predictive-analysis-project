clear;
close all;
clc;

data = readtable('data.csv');
data(2,:); %Pour avoir la 2eme ligne en entier
data(:,2); %Pour avoir la 2eme colonne en entier
t=0:1:206;
KOR=0*t;AUS=0*t;ISL=0*t;GBR=0*t;EST=0*t;

for k = 1:6408
    if string(data{k,1}) == 'KOR'
        KOR(k) = string(data{k,7});
    elseif string(data{k,1}) == 'AUS'
        AUS(k) = string(data{k,7});
    elseif string(data{k,1}) == 'ISL'
        ISL(k) = string(data{k,7});
    elseif string(data{k,1}) == 'GBR'
        GBR(k) = string(data{k,7});
    elseif string(data{k,1}) == 'EST'
        EST(k) = string(data{k,7});        
    end
end