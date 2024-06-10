% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for quantization with GATE discrimination
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
% *** Floating Point Representations ***
% FP32 = 1/8/23 (Single-precision floating-point format)
% FP16 = 1/5/10 (Half-precision floating-point format)
% Bfloat16 = 1/8/7 (bfloat16 floating-point format)
% FP8  = 1/4/3 (Minifloat)
%
% *** Integer Representations ***
% INT16, INT8, INT4, INT2
%
% https://nhigham.com/2020/05/26/simulating-low-precision-floating-point-arithmetics-in-matlab/
% https://github.com/higham/chop
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Deadzone quantizer
% The most common form of nonuniform quantizer is the deadzone quantizer.
% This has a broader decision range for the band of inputs close to zero.
% It has the benefit during compression of ensuring that noisy low-level
% signals are not allocated bits unnecessarily.
% *************************************************************************


%% Initializes the Enviornment
clc
clear
close all

%% Data Load
load('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch15CellSize100.mat');
load('SisFall_SCALE2FILTER1DOWN0_SEP2022.mat')

% Structure the Data
lstm_I = net.Layers(2).InputWeights;
lstm_W = net.Layers(2).RecurrentWeights;
lstm_B = net.Layers(2).Bias;
fc_B   = net.Layers(3).Bias;
fc_W   = net.Layers(3).Weights;

%% Bias Analysis
figure

% Bias Plot
xx = 1:1:length(lstm_B);
plot(xx,lstm_B,'-.or')
hold on
ylim([-0.2 1.2])
% xlim([0 120])
grid

% Zone Rectangles
x = [0 99 99 0];
y = [-0.2 -0.2 1.2 1.2];
patch(x,y,'b','FaceAlpha',0.1)
x = [99 200 200 99];
y = [-0.2 -0.2 1.2 1.2];
patch(x,y,'g','FaceAlpha',0.1)
x = [200 400 400 200];
y = [-0.2 -0.2 1.2 1.2];
patch(x,y,'b','FaceAlpha',0.1)

% Arrows
x = [0.13 ,0.32];
y = [0.5 ,0.5];
a = annotation('doublearrow',x,y);
t = text(30,0.4,'Forget Gate','Rotation',45);

x = [0.32 ,0.52];
y = [0.5 ,0.5];
a = annotation('doublearrow',x,y);
t = text(130,0.4,'Input Gate','Rotation',45);

x = [0.51 ,0.71];
y = [0.5 ,0.5];
a = annotation('doublearrow',x,y);
t = text(230,0.4,'Output Gate','Rotation',45);

xline(300,'--k')

x = [0.71 ,0.89];
y = [0.5 ,0.5];
a = annotation('doublearrow',x,y);
t = text(330,0.4,'Cell Gate','Rotation',45);


%% INT
b = 16;                      % Number of bits for the Quantization
P=0;                        % Draws the Quantization Lines

lstm_I = net.Layers(2).InputWeights;
lstm_W = net.Layers(2).RecurrentWeights;
lstm_B = net.Layers(2).Bias;
fc_B = net.Layers(3).Bias;
fc_W = net.Layers(3).Weights;

[lstm_I_Q,Si,Z] = Quantize(lstm_I,b);
[lstm_W_Q,Sw,Z] = Quantize(lstm_W,b);

N = length(lstm_B);
N4 = N/4;
f_lstm_B = lstm_B(   1:  N4);
i_lstm_B = lstm_B(  N4+1:2*N4);
o_lstm_B = lstm_B(2*N4+1:3*N4);
c_lstm_B = lstm_B(3*N4+1:4*N4);

[f_lstm_B_Q,fSb,Z] = Quantize(f_lstm_B,b);
[i_lstm_B_Q,iSb,Z] = Quantize(i_lstm_B,b);
[o_lstm_B_Q,oSb,Z] = Quantize(o_lstm_B,b);
[c_lstm_B_Q,cSb,Z] = Quantize(c_lstm_B,b);

[fc_B_Q,Sfb,Z]  = Quantize(fc_B,b);
[fc_W_Q,Sfw,Z]  = Quantize(fc_W,b);

lstm_I_Q_r = deQuantize(lstm_I_Q,Si,Z);
lstm_W_Q_r = deQuantize(lstm_W_Q,Sw,Z);

f_lstm_B_Q_r = deQuantize(f_lstm_B_Q,fSb,Z);
i_lstm_B_Q_r = deQuantize(i_lstm_B_Q,iSb,Z);
o_lstm_B_Q_r = deQuantize(o_lstm_B_Q,oSb,Z);
c_lstm_B_Q_r = deQuantize(c_lstm_B_Q,cSb,Z);
lstm_B_Q_r = [f_lstm_B_Q_r; i_lstm_B_Q_r; o_lstm_B_Q_r; c_lstm_B_Q_r];

fc_B_Q_r   = deQuantize(fc_B_Q,Sfb,Z);
fc_W_Q_r   = deQuantize(fc_W_Q,Sfw,Z);

plot(xx,lstm_B_Q_r,'-.sb')

% Draws the Quantization Lines
if P==1
    mf = min(min(f_lstm_B));
    mi = min(min(i_lstm_B));
    mo = min(min(o_lstm_B));
    mc = min(min(c_lstm_B));
    for i=0:2^b
        Af = mf+i*fSb-fSb/2; pf = Af*ones(1,length(xx)/4);
        Ai = mi+i*iSb-iSb/2; pi = Ai*ones(1,length(xx)/4);
        Ao = mo+i*oSb-oSb/2; po = Ao*ones(1,length(xx)/4);
        Ac = mc+i*cSb-cSb/2; pc = Ac*ones(1,length(xx)/4);
        p = [pf pi po pc];
        plot(xx,p,'--m','LineWidth',1);
    end
end


r = [reshape(lstm_I,1,[])'; reshape(lstm_W,1,[])'; reshape(lstm_B,1,[])'; ...
    reshape(fc_B,1,[])'; reshape(fc_B,1,[])'];
Q = [reshape(lstm_I_Q_r,1,[])'; reshape(lstm_W_Q_r,1,[])'; reshape(lstm_B_Q_r,1,[])'; ...
    reshape(fc_B_Q_r,1,[])'; reshape(fc_B_Q_r,1,[])'];

RMSE = sqrt(mean((r - Q).^2)/length(r))




%% Update NET

% Calculates INITIAL the Confusion Matrix
figure;

YPred_V = classify(net,XV);

cm = confusionchart(YV,YPred_V);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Extracts each cell value
TP = cm.NormalizedValues(1,1);
FN = cm.NormalizedValues(1,2);
FP = cm.NormalizedValues(2,1);
TN = cm.NormalizedValues(2,2);

% Metrics Calculation
Accuracy = (TP+TN)/(TP+TN+FP+FN)

% Replaces the Layers
tmp_net = net.saveobj;
tmp_net.Layers(2).InputWeights = lstm_I_Q_r;
tmp_net.Layers(2).RecurrentWeights = lstm_W_Q_r;
tmp_net.Layers(2).Bias = lstm_B_Q_r;
tmp_net.Layers(3).Bias = fc_B_Q_r;
tmp_net.Layers(3).Weights = fc_W_Q_r;

net = net.loadobj(tmp_net);

YPred_V_Q = classify(net,XV);

% Calculates the QUANTIZED Confusion Matrix
f = figure;
cm = confusionchart(YV,YPred_V_Q);
% cm.RowSummary = 'row-normalized';
% cm.ColumnSummary = 'column-normalized';

% Extracts each cell value
TP = cm.NormalizedValues(1,1);
FN = cm.NormalizedValues(1,2);
FP = cm.NormalizedValues(2,1);
TN = cm.NormalizedValues(2,2);

% Metrics Calculation
Accuracy     = (TP+TN)/(TP+TN+FP+FN)

File = strcat('NetQuantization_GATE_b',num2str(b),'.png');
% saveas(f,File);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%