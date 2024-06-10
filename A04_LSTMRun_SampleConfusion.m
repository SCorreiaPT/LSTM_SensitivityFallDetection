% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for inference a LSTM for Fall Detection
% using the "classify" MatLab function
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
clear
close all
clc

%% Data Load
load('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch15CellSize100.mat')
load('SisFall_SCALE2FILTER1DOWN0_SEP2022.mat');

%% Classifies the all dataset
YPred_T = classify(net,XT);         % Using Training Data (not adequate)
YPred_V = classify(net,XV);         % Using Validation Data

YPred = classify(net,XV{1});        % Only one sample

figure
plot(YPred_T,'o')
title('Using Training Data')
xlabel('Sample')
figure
plot(YPred_V,'o')
title('Using Validation Data')
xlabel('Sample')


%% Evaluation

% Calculates the Confusion Matrix
cm = confusionchart(YV,YPred_V);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Extracts each cell value
TP = cm.NormalizedValues(1,1);
FN = cm.NormalizedValues(1,2);
FP = cm.NormalizedValues(2,1);
TN = cm.NormalizedValues(2,2);

% Metrics Calculation
Accuracy     = (TP+TN)/(TP+TN+FP+FN);
%Accuracy = mean(YT == YPred)

F1 = 2*TP/(2*TP+FP+FN);

Sensitivity  = TP/(TP+FN);
Specificity  = TN/(TN+FP);
Precision    = TP/(TP+FP);
NegPredValue = TN/(TN+FN);

% *************************************************************************