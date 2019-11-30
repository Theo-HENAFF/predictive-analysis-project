clear;
close all;
clc;

data = readtable('data_KOR.csv');
data(2,:); %Pour avoir la 2eme ligne en entier
data(:,2); %Pour avoir la 2eme colonne en entier
KOR_Values = data.Value;
KOR_1968_2015 = KOR_Values(1:184);

K=27;
d=0; %d pour decalage
%AR : 


modele_ar = ar(KOR_1968_2015,K);
modele_armax = armax(KOR_1968_2015,[K,K]);
prediction_AR = forecast(modele_ar,KOR_1968_2015,22);
prediction_ARMAX = forecast(modele_armax,KOR_1968_2015,22);

x=(1:206);


subplot(1,2,1);
plot(x(185-d:206-d),prediction_AR,'r');
hold on
plot(KOR_Values,'g');
plot(KOR_1968_2015,'b');
hold off
title('AR');

%ARMAX : autour de 25 à -+ 2 c'esp pas trop trop mal 26 et 27 ok
subplot(1,2,2);
plot(x(185-d:206-d),prediction_ARMAX,'r');
hold on
plot(KOR_Values,'g');
plot(KOR_1968_2015,'b');
hold off
title('ARMAX');


