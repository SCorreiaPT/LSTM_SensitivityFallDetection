% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for inference a LSTM for Fall Detection
% using the "predictAndUpdateState" MatLab function
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
load('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_epoch1.mat')
load('SisFall_SCALE2FILTER1DOWN0_SEP2022.mat');

[updatedNet,YPred] = predictAndUpdateState(net,XV);
[updatedNet,YPred_Class] = predictAndUpdateState(net,XV,'ReturnCategorical',1);

plot(YPred,'o')
title('Using Validation Data')
xlabel('Sample')

acc = mean(YV == YPred_Class);

cm = confusionchart(YV,YPred_Class);
cm.Title = 'Using Validation Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

TP = cm.NormalizedValues(1,1);
FN = cm.NormalizedValues(1,2);
FP = cm.NormalizedValues(2,1);
TN = cm.NormalizedValues(2,2);

Accuracy     = (TP+TN)/(TP+TN+FP+FN);
F1 = 2*TP/(2*TP+FP+FN);

Sensitivity  = TP/(TP+FN);
Specificity  = TN/(TN+FP);
Precision    = TP/(TP+FP);
NegPredValue = TN/(TN+FN);

% *************************************************************************