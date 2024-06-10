% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for a LSTM Network calibration 
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
% - Histograms for each layer
% *************************************************************************

%% Initializes the Enviornment
clc
clear
close all

%% Data Load
load('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch15CellSize100.mat');

% Structure the Data
lstm_I = net.Layers(2).InputWeights;
lstm_W = net.Layers(2).RecurrentWeights;
lstm_B = net.Layers(2).Bias;
fc_W   = net.Layers(3).Weights;
fc_B   = net.Layers(3).Bias;

%% Input Weights 
figure

% Histogram
yyaxis left
Hist_I = histogram(lstm_I,16);
ylabel('Number of Occurrences') 
hold on

% Normal Distribuition Equivalent
yyaxis right
[sigma,mu] = std(lstm_I,0,'all');
y = Hist_I.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')


% Mean, Standard Deviation Annotation
dim = [.68 .58 .3 .3];
str = {strcat('$\mu=',num2str(mu),'$'),strcat('$\sigma=',num2str(sigma),'$')};
a = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex','Color',[0.8500 0.3250 0.0980]);


%% Recurrent Weights
figure

% Histogram
yyaxis left
Hist_W = histogram(lstm_W,26);
ylabel('Number of Occurrences') 

% Normal Distribuition Equivalent
yyaxis right
[sigma,mu] = std(lstm_W,0,'all');
y = Hist_W.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

% Mean, Standard Deviation Annotation
dim = [.68 .58 .3 .3];
str = {strcat('$\mu=',num2str(mu),'$'),strcat('$\sigma=',num2str(sigma),'$')};
a = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex','Color',[0.8500 0.3250 0.0980]);

%% Bias
figure

% Histogram
yyaxis left
Hist_B = histogram(lstm_B,15);
ylabel('Number of Occurrences')

% Normal Distribuition Equivalent
yyaxis right
[sigma,mu] = std(lstm_B,0,'all');
y = Hist_B.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')
grid

% Mean, Standard Deviation Annotation
dim = [.68 .58 .3 .3];
str = {strcat('$\mu=',num2str(mu),'$'),strcat('$\sigma=',num2str(sigma),'$')};
a = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex','Color',[0.8500 0.3250 0.0980]);


%% Bias Analysis
figure

% Bias Plot
x = 1:1:length(lstm_B);
plot(x,lstm_B,'-.or')
ylim([-0.2 1.2])
xlim([0 400])
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


%% Fully Connected Weights
figure

% Histogram
yyaxis left
Hist_fc = histogram(fc_W,8);
ylabel('Number of Occurrences') 

% Normal Distribuition Equivalent
yyaxis right
[sigma,mu] = std(fc_W,0,'all');
y = Hist_fc.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

% Mean, Standard Deviation Annotation
dim = [.68 .58 .3 .3];
str = {strcat('$\mu=',num2str(mu),'$'),strcat('$\sigma=',num2str(sigma),'$')};
a = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex','Color',[0.8500 0.3250 0.0980]);


%% Fully Connected Bias
figure

% Histogram
yyaxis left
Hist_fcB = histogram(fc_B);
ylabel('Number of Occurrences') 

% Normal Distribuition Equivalent
yyaxis right
[sigma,mu] = std(fc_B,0,'all');
y = Hist_fcB.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

%% All in One Histogram
ds = 'stairs';
ds = 'bar';
figure
Hist_I   = histogram(lstm_I,16,'Normalization','pdf','displaystyle',ds);
hold on
Hist_W   = histogram(lstm_W,26,'Normalization','pdf','displaystyle',ds);
Hist_B   = histogram(lstm_B,10,'Normalization','pdf','displaystyle',ds);
Hist_fcW = histogram(fc_W,8,'Normalization','pdf','displaystyle',ds);
Hist_fcB = histogram(fc_B,'displaystyle',ds);

legend('lstm_I','lstm_W','lstm_B','fc_W','fc_B');
grid

%% Gates Histograms
forget = lstm_B( 1: 29);
figure;
yyaxis left
h=histogram(forget,16);
ylabel('Number of Occurrences') 
yyaxis right
[sigma,mu] = std(forget,0,'all');
M = max(forget);
m = min(forget);
y = h.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

input  = lstm_B(30: 58);
figure;
yyaxis left
h=histogram(input,16);
ylabel('Number of Occurrences') 
yyaxis right
[sigma,mu] = std(input,0,'all');
M = max(input);
m = min(input);
y = h.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

output = lstm_B(59: 87);
figure;
yyaxis left
h=histogram(output,16);
ylabel('Number of Occurrences') 
yyaxis right
[sigma,mu] = std(output,0,'all');
M = max(output);
m = min(output);
y = h.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

cell   = lstm_B(88:116);
figure
yyaxis left
h=histogram(cell,16);
ylabel('Number of Occurrences') 
yyaxis right
[sigma,mu] = std(cell,0,'all');
M = max(cell);
m = min(cell);
y = h.BinEdges;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)
ylabel('Probability Density')

% *************************************************************************