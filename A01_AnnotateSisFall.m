% *************************************************************************
% MatLab script for reading the SisFall dataset and creating a MatLab
% data file. The imported dataset is annotaded with a new column and the
% second set of accelerometer data is removed. 
%
% SisFall: A Fall and Movement Dataset
% Created by:
% A. Sucerquia, J.D. López, J.F. Vargas-Bonilla
% SISTEMIC, Faculty of Engineering, Universidad de Antiquia UDEA
% February 2016 - Version 1.0
% http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
% 
% Annotations Created by:
% Sérgio Correia and João P. Matos-Carvalho, December 2022
% Laboratory of Electronics and Instrumentation, Advanced Computing 
% Technologies and Applications
% Instituto Politécnico de Portalegre, Escola Superior Tecnologia e Gestão
% Portalegre, Portugal
%
% Notes
% - a windows of 2s is created, centered on the fall event
% - from the 5 trial of the dataset, 1 is considered for validation
% *************************************************************************


% Initializes the IDE Environment
clear 
close all
clc

fs = 200;                   % Sampling frequency
dt = 2;                     % Event considered interval

D = 1:19;                   % Activities of Daily Living (ADL)
F = 1:15;                   % Falls

SA = 1:23;                  % SA: Adults subjects between 19 and 30 years old
SE = 1:15;                  % SE: Elderly people between 60 and 75 years old

R = 1:5;                    % Trial Number 

fc = 5;                     % Filtering data (LOW-PASS)

FILTER = 1;                 % Control Variables - Low pass filter
SCALE = 2;                  % Control Variables - 1 = Rescale to [0, 1]
                            %                   - 2 = Rescale to [-1,1]
                            %                   - 3 = Normalize

DOWN = 1;                   % Control Variables - Downsample input data
                            % DOWN = 1 -> No changes
                            % 10(20Hz) 9(22Hz) 8(25Hz) 7(28Hz) 6(33Hz) 5(40Hz) 4(50Hz) 3(66Hz) 2(100Hz)

OUTPUT = 1;                 % Control Variables - Output
                            %                   - 1 = categorical cell
                            %                   - 2 = vector cell
FORCE_WINDOW_ADL = 4;       % Control Variables - FORCE_WINDOW_ADL
                            %                   - 4 = 1/4 window as ADL
                            %                   - 3 = 1/3 window as ADL
%% ************************************************************************
% Classifies 1.725 FALLs
% 15 Code Activityies, 23 Adults Subjects, 5 Tries (15x23x5)
% 1.380 for Training (80%)
%   345 for Validation (20%)
% Shortens the data frame to a 2s window centered on the fall event
tt=1;
tv=1;
XT={};XV={};

if OUTPUT == 1
    YT = categorical.empty;
    YV = categorical.empty;
else
    YT={};
    YV={};
end



for i=1:15;         % Falls
    for j=1:23      % SA Adults subjects
        for w=1:5   % Tries
    
            % Build the Filename
            if i<10; z1='0';else z1='';end;
            if j<10; z2='0';else z2='';end;
            if w<10; z3='0';else z3='';end;

            mydir = pwd;
            idcs = strfind(mydir, '/');
            newdir = mydir(1:idcs(end));

            f = strcat(newdir,'data/SisFall_dataset/','F',z1,num2str(D(i)),'_','SA',z2,num2str(SA(j)),'_','R',z3,num2str(R(w)),'.txt');
            
            % Reads data file and scales
            data = readmatrix(f);
            Acc = [(2.*16)/(2^13)]*data(:,1:3);  % ADXL345 Accelerometer
                                                 % +/-16g scale
                                                 % 13 bits resolution
            % Downsample the input data
            Acc = downsample(Acc,DOWN);	         
            fss=fs/DOWN;
            wd=floor(fss*dt/2);

            % Looks for the fall event
            [M,I] = max(abs(Acc));
            [MM,II] = max(M);
            I = I(II);

            [tt, tv, XT, YT, XV, YV] = run(fc, fss, Acc, wd, I, 'FALL', FILTER, SCALE, OUTPUT, w, tt, tv, XT, YT, XV, YV, FORCE_WINDOW_ADL);           
        end %(w=1:5)
    end %(j=1:23)
end %(i=1:15)



%% ************************************************************************
% Classifies +/- 1.812 ADLs
% 19 Code Activityies, 23 Adults Subjects, 5 Tries (15x23x5)
% Shortens the data frame to a 2s window centered on middle of the interval
R = [1 1 1 1 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5];    % Number of Tries
for i=1:19                                      % 19 Daily Activities (ADL)
    for j=1:23                                  % 23 Adults subjects (SA)     
        for w=1:R(i)

            % Build the Filename
            if i<10; z1='0';else z1='';end;
            if j<10; z2='0';else z2='';end;
            if w<10; z3='0';else z3='';end;

            mydir = pwd;
            idcs = strfind(mydir, '/');
            newdir = mydir(1:idcs(end));

            f = strcat(newdir,'data/SisFall_dataset/','D',z1,num2str(D(i)),'_','SA',z2,num2str(SA(j)),'_','R',z3,num2str(R(w)),'.txt');
            
            % Reads data file and scales
            data = readmatrix(f);
            Acc = [(2.*16)/(2^13)]*data(:,1:3);

            % Downsample the input data
            Acc = downsample(Acc,DOWN);
            fss=fs/DOWN;
            wd=floor(fss*dt/2);

            % Centers the interval
            I = round(length(Acc)/2);

            [tt, tv, XT, YT, XV, YV] = run(fc, fss, Acc, wd, I, 'ADL', FILTER, SCALE, OUTPUT, w, tt, tv, XT, YT, XV, YV, FORCE_WINDOW_ADL);         
        end
    end
end

%% ************************************************************************
% Saves training and validation data on a MatLab variable file
f = strcat('SisFall_','SCALE',num2str(SCALE),'FILTER',num2str(FILTER),'DOWN',num2str(DOWN),'dt',num2str(dt),'OUTPUT',num2str(OUTPUT), 'FORCE_WINDOW_ADL', num2str(FORCE_WINDOW_ADL), '_',date,'.mat');
save(f,'XT','YT','XV','YV','SCALE','FILTER','DOWN','OUTPUT');

function [tt, tv, XT, YT, XV, YV] = run(fc, fss, Acc, wd, I, CLASS, FILTER, SCALE, OUTPUT, w, tt, tv, XT, YT, XV, YV, FORCE_WINDOW_ADL)
    % Filtering data (LOW-PASS)
    % Butterworth, Order=4, fs=200Hz, CutOff=5Hz
    if FILTER==1
        [b,a] = butter(4,fc/(fss/2));% Butterworth, Order=4, fs=200Hz, CutOff=5Hz
        Acc = filter(b,a,Acc);
    end

    % Centers a 2s data window (f=200Hz)
    Sl = I-wd;
    Sr = I+wd;
    if Sl < 1
        Sl = 1;
        Sr = wd * 2 + 1;
    elseif Sr > length(Acc)
        Sl = length(Acc) - wd * 2;
        Sr = length(Acc) + 1;
    end
    A = Acc(Sl:Sr-1,:);	
   
    % Normalization Rescaling Values [0,1]
    if SCALE==1
        A = rescale(A);
    end
    % Normalization Rescaling Values [-1,1]
    if SCALE==2
        A = rescale(A,-1,1);
    end
    % Normalization Rescaling Values (Acc-mu)/sigma
    % Zero mean and unity standard deviation distribuition
    if SCALE==3
        mu = mean(A);
        sigma = std(A);
        A = (A-mu)./sigma;
    end    

    % Saves data vector (20% of total data size)
    if w==5
        % Validation data
        XV(tv,1) = {A'};

        if OUTPUT == 1
            YV(tv,1) = categorical(cellstr(CLASS));
        else
            tmp = repmat({CLASS},length(Sl:Sr-1),1)';
            tmp(1,1:floor(length(tmp)/FORCE_WINDOW_ADL)) = {'ADL'};
            YV(tv,1) = {categorical(tmp)};
            %YV(tv,1) = {categorical(repmat({CLASS},length(Sl:Sr-1),1)')};
        end
        % Vetor position increment
        tv = tv + 1    
    else
        % Train data
        XT(tt,1) = {A'};

        if OUTPUT == 1
            YT(tt,1) = categorical(cellstr(CLASS));
        else
            tmp = repmat({CLASS},length(Sl:Sr-1),1)';
            tmp(1,1:floor(length(tmp)/FORCE_WINDOW_ADL)) = {'ADL'};
            YT(tt,1) = {categorical(tmp)};
            %YT(tt,1) = {categorical(repmat({CLASS},length(Sl:Sr-1),1)')};
        end
        % Vetor position increment
        tt = tt + 1    
    end %(w==5)  
end
