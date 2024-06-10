% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for implementing the memory model of a 1 layer
% LSTM Network for fall detections
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

%% Initializes the Enviornment
clear
clc
close all

%% Considered Processors

Stack = 0.75;
% MICROCHIP MSP430 512KB Flash, 66KB RAM
Z_MPS = 66000*Stack;

% Silicon Labs C8051F98x 8KB Flash, 512 KB RAM
Z_Si = 512000*Stack;

% ST STM88L 64 KB Flash, 6 KB RAM
Z_ST8 = 6000*Stack;

% ST STM32L 1MB Flash, 320 KB RAM
Z_ST32 = 320000*Stack;

% LSTM Layers 1 and 2 unit size
[h1,h2] = meshgrid(10:10:200,10:10:200);

% Representation Size
Sz = 4;

% Number of inputs
d=3;

%% Memory Model
Mem = Sz*4.*h1 + Sz*4.*h1*d + Sz*4.*h1.^2 + ...
    + Sz*4.*h2 + Sz*4.*h2*d + Sz*4.*h2.^2 + ...
    + Sz*2.*h2 + Sz.*2;

Sz = 2;
Mem2 = Sz*4.*h1 + Sz*4.*h1*d + Sz*4.*h1.^2 + ...
    + Sz*4.*h2 + Sz*4.*h2*d + Sz*4.*h2.^2 + ...
    + Sz*2.*h2 + Sz.*2;

Sz = 1;
Mem3 = Sz*4.*h1 + Sz*4.*h1*d + Sz*4.*h1.^2 + ...
    + Sz*4.*h2 + Sz*4.*h2*d + Sz*4.*h2.^2 + ...
    + Sz*2.*h2 + Sz.*2;

Sz = 0.5;
Mem4 = Sz*4.*h1 + Sz*4.*h1*d + Sz*4.*h1.^2 + ...
    + Sz*4.*h2 + Sz*4.*h2*d + Sz*4.*h2.^2 + ...
    + Sz*2.*h2 + Sz.*2;

Sz = 0.25;
Mem5 = Sz*4.*h1 + Sz*4.*h1*d + Sz*4.*h1.^2 + ...
    + Sz*4.*h2 + Sz*4.*h2*d + Sz*4.*h2.^2 + ...
    + Sz*2.*h2 + Sz.*2;


surf(h1,h2,Mem)
hold on
z1 = Z_MPS*ones(length(h1));
surf(h1,h2,z1,'FaceAlpha',0.1)

z2 = Z_Si*ones(length(h1));
surf(h1,h2,z2,'FaceAlpha',0.1)


xlabel('LSTM 2');
ylabel('LSTM 1');
zlabel('Memory Occupancy');

dim = [.55 .01 .3 .25];
str = 'MPS430';
annotation('textbox',dim,'String',str,'FitBoxToText','on',Color='r');

dim = [.75 .15 .3 .3];
str = 'C8051F98x';
annotation('textbox',dim,'String',str,'FitBoxToText','on',Color='r');

% Adjusts Axis
% view(3); camlight; axis vis3d

% Claculates the interception line 1
zdiff = Mem - z1;
C = contours(h1, h2, zdiff, [0 0]);

xL = C(1, 2:end);
yL = C(2, 2:end);

zL = interp2(h1, h2, z1, xL, yL);

line(xL, yL, zL, 'Color', 'r', 'LineWidth', 3);

% Claculates the interception line 2
zdiff = Mem - z2;
C = contours(h1, h2, zdiff, [0 0]);

xL = C(1, 2:end);
yL = C(2, 2:end);

zL = interp2(h1, h2, z2, xL, yL);

line(xL, yL, zL, 'Color', 'r', 'LineWidth', 3);

% *************************************************************************
