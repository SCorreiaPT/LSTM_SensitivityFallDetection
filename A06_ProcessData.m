% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for processing LSTM inference resultls from 
% multiple 1 Layer networks for Fall Detection
% *************************************************************************
%
% SisFall: A Fall and Movement Dataset
% Created by:
% A. Sucerquia, J.D. López, J.F. Vargas-Bonilla
% SISTEMIC, Faculty of Engineering, Universidad de Antiquia UDEA
% February 2016 - Version 1.0
% http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
% 
% Script Created by:
% Sérgio Correia, November 2022
% Laboratory of Electronics and Instrumentation, Advanced Computing 
% Technologies and Applications
% Instituto Politécnico de Portalegre, Escola Superior Tecnologia e Gestão
% Portalegre, Portugal
%
% Notes
% - 
% *************************************************************************

%% Inicializations

close all
clear
clc

%% Data Load
load('SisFall_SCALE2FILTER1DOWN0_SEP2022.mat');

% Index inicialization
j = 1;

% Cell Size to Load
x1 = 1:1:29;
x2 = 30:10:200;
x= [x1 x2];

for w=1:length(x)   % 67

    % Cell Size
    i = x(w);

    % Network File
    f = strcat('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch15CellSize',num2str(i),'.mat');
    load(f);
    
    % Classification
    YPred_V = classify(net,XV);
    acc = mean(YV == YPred_V);
    cm = confusionchart(YV,YPred_V);
    
    % Metrics
    TP = cm.NormalizedValues(1,1);
    FN = cm.NormalizedValues(1,2);
    FP = cm.NormalizedValues(2,1);
    TN = cm.NormalizedValues(2,2);
    
    Precision(j)    = TP/(TP+FP);
    Sensibility(j)  = TP/(TP+FN);
    Specificity(j)  = TN/(TN+FP);
    NegPredValue(j) = TN/(TN+FN);
    
    Accuracy(j) = (TP+TN)/(TP+FN+FP+TN);
    F1(j) = 2*TP/(2*TP+FP+FN);
    Loss(j) = info.FinalValidationLoss;

    % Increment Index
    j= j + 1
end

% Saves Global Result
save('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch10_ALL_V.mat','F1','x','Loss','Accuracy','Precision','Sensibility','Specificity','NegPredValue');

% Plots Global Result
plot(x,F1,'-*m',x,Loss,'-oy',x,Accuracy,'-ok',x,Precision,'-*r',x,Sensibility,'-dg',x,Specificity,'-+b',x,NegPredValue,'-xm');
grid
ylim([0 1])
legend('F1','Loss','Accuracy','Precision','Sensibility','Specificity','Negative Preditive Value','Location','southeast');

% *************************************************************************

