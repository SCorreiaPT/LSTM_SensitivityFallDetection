% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for training a 2 Layer LSTM for Fall Detection
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
% Sérgio Correia and João P. Matos-Carvalho, December 2022
% Laboratory of Electronics and Instrumentation, Advanced Computing 
% Technologies and Applications
% Instituto Politécnico de Portalegre, Escola Superior Tecnologia e Gestão
% Portalegre, Portugal
%
% Notes
% - the created and trained network is stored in a *.mat file
% *************************************************************************

%% Inicializations
clear
close all
clc

%% Dataset for training
File = 'SisFall_SCALE1FILTER1DOWN1_28-Nov-2022';
load(strcat('results/A01/',File,'.mat'));

%% Training Options
miniBatchSize = 31;         % 2852 Samples, with 31 Samples for the Batch Size -> 92 Iterations
Epochs = 15;                % The Accuracy should be stable by this value
ValidationFrequency = 46;   % 92 Iterations, with 46 Validation Interval -> 2 Validations per Epoch
options = trainingOptions('adam', ...
    'ValidationFrequency', ValidationFrequency, ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',Epochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XV,YV}, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');


%% Create/Train/Save the network
CellSizeL = 1:1:29;
CellSizeH = 30:10:400;
CellSizeLH = [CellSizeL CellSizeH];

if ~exist('results/A02_2', 'dir')
    mkdir('results/A02_2')
end

for C1 = 1:1:length(CellSizeLH)  % 67
    for C2 = 1:1:length(CellSizeLH)
        
        CellSize1 = CellSizeLH(C1);
        CellSize2 = CellSizeLH(C2);
        MinLength=400;
        layers = [
            sequenceInputLayer(3,"Name","SequenceInput","MinLength",MinLength)
            lstmLayer(CellSize1,"Name","LSTM_1","OutputMode","last")
            lstmLayer(CellSize2,"Name","LSTM_2","OutputMode","last")
            fullyConnectedLayer(2,"Name","FullConnected_1")
            softmaxLayer("Name","SoftMax")
            classificationLayer("Name","ClassificationOutput")];
        % plot(layerGraph(layers));
    
        % Train the network
        [net,info] = trainNetwork(XT,YT,layers,options);
    
        % Closes pending figure
        h= findall(groot,'Type','Figure');
        close(h);
    
        % Saves training and validation data on a MatLab variable file
        FName = strcat('results/A02_2/','TainedNET_SisFall_SCALE1FILTER1DOWN1','_Epoch',num2str(Epochs, '%04.f'),'CellSize',num2str(CellSize1,'%04.f'),'x',num2str(CellSize2, '%04.f'),'.mat');
        save(FName,'net','info');
    end
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%