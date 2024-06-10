% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatLab script for 1 Layer LSTM Fall Detection Evaluation
% SCorreia, October 2022
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Accuracy(j) = (TP+TN)/(TP+FN+FP+TN);
%     F1(j) = 2*TP/(2*TP+FP+FN);
%
% F1 score is balancing precision and recall on the positive class 
% while accuracy looks at correctly classified observations both 
% positive and negative
% Harmonic mean of Precision and Recall
%
%     Precision(j)    = TP/(TP+FP); % How many of those who were labeled as FALL are actually FALLs?
%     ->Recall(j)     = TP/(TP+FN); % Of all the samples that are FALLs, how many of those were correctly predicted?
%     Specificity(j)  = TN/(TN+FP); % Of all the samples who are ADLs, how many of those were correctly predicted?
%     NegPredValue(j) = TN/(TN+FN); % How many of those who were labeled as ADLs are actually ADLs?
%
% Recall is how sure you are that you are not missing any positives.
% Choose Recall if the idea of false positives is far better than false negatives, in other words, if the occurrence of false negatives is unaccepted/intolerable, that you’d rather get some extra false positives(false alarms) over saving some false negatives, like in our diabetes example.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Enviornement Evaluation
clc
clear
close all

%% Load Summarized Data
load('TainedNET_SisFall_SCALE2FILTER1DOWN0_SEP2022_Epoch10_ALL_V.mat');

% Maximum and Minimum values, not considering initial values
[AccMAX,idxMAX] = max(Accuracy);                % CellSize = 100 (97,25%)
[AccMIN,idxMIN] = min(Accuracy(7:length(x)));   % CellSize = 26  (93,91%)

%% Figure will all the data 
% figure
% plot(x,Accuracy,'-ok',x,Precision,'-*r',x,Sensibility,'-dg',x,Specificity,'-+b',x,NegPredValue,'-xm');
% grid
% ylim([0.5 1])
% lgd = legend('Accuracy','Precision','Sensibility','Specificity','Negative Preditive Value','Location','southoutside');
% lgd.NumColumns  = 2;

%% Accuracy Analysis
figure

iL = 1:10:29;
iH = 30:1:47;
i = [iL iH];

plt = plot(x,Accuracy,'-db',x,Loss,'-.r','LineWidth',2,'MarkerIndices',i);
hAx=gca;  % avoid repetitive function calls
set(hAx,'xminorgrid','on','yminorgrid','on')

grid
ylim([0 1])
lgd = legend('Accuracy','Loss','','','Location','southoutside');
lgd.NumColumns  = 2;

yline(AccMAX,'r','HandleVisibility','off')
yline(AccMIN,'r','HandleVisibility','off')

%% Zoom Window with Annotations
figure
plt = plot(x(1:40),Accuracy(1:40),'-db','LineWidth',2,'MarkerIndices',i);
hAx=gca;  % avoid repetitive function calls
set(hAx,'xminorgrid','on','yminorgrid','on')
ylim([0.7 1])

yline(AccMAX,'r')
yline(AccMIN,'r')

text(120,AccMAX+0.005,strcat(num2str(round(AccMAX*100,2)),'%'),'Color','r')
text(120,AccMIN-0.005,strcat(num2str(round(AccMIN*100,2)),'%'),'Color','r')

dAcc = round((AccMAX-AccMIN)*100,2);
a = annotation('textarrow',[.65 .74],[AccMAX-0.25 AccMAX-0.15],'String',strcat(num2str(dAcc),'%'));
a.Color = 'red';

a = annotation('doublearrow',[.75 .75],[AccMAX-0.12 AccMAX-0.21]);
a.Color = 'red';

%% Figure will Accuracy, Precision and Recall 
figure
plot(x,Accuracy,'-.or','LineWidth',1);
hold on
plot(x,Precision,'-hg','LineWidth',2,'color',[0.4660 0.6740 0.1880]);
plot(x,Sensibility,'-db','LineWidth',2);
grid
ylim([0 1])
lgd = legend('Accuracy','Precision','Recall','Location','southoutside');
lgd.NumColumns  = 3;

hAx=gca;  % avoid repetitive function calls
set(hAx,'xminorgrid','on','yminorgrid','on')

% Zoom
figure
plot(x(1:40),Accuracy(1:40),'-.or','LineWidth',1);
hold on
plot(x(1:40),Precision(1:40),'-hg','LineWidth',2,'color',[0.4660 0.6740 0.1880]);
plot(x(1:40),Sensibility(1:40),'-db','LineWidth',2);
grid
ylim([0.8 1])
lgd = legend('Accuracy','Precision','Recall','Location','southoutside');
lgd.NumColumns  = 3;

hAx=gca;  % avoid repetitive function calls
set(hAx,'xminorgrid','on','yminorgrid','on')


%% Memory Footprint
Inputs = 3;
Cells = x;
% Cells = 100;

inWeights = 4.*Cells*Inputs;
ReWeights = 4.*Cells.*Cells;
BiWeights = 4.*Cells;
FCWeights = 2.*Cells;
FCBias    = 2;

VarSize = 4;
Mem = VarSize.*(inWeights+ReWeights+BiWeights+FCWeights+FCBias);

VarSize = 2;
Mem_16 = VarSize.*(inWeights+ReWeights+BiWeights+FCWeights+FCBias);

VarSize = 1;
Mem_8 = VarSize.*(inWeights+ReWeights+BiWeights+FCWeights+FCBias);

VarSize = 0.5;
Mem_4 = VarSize.*(inWeights+ReWeights+BiWeights+FCWeights+FCBias);

VarSize = 0.25;
Mem_2 = VarSize.*(inWeights+ReWeights+BiWeights+FCWeights+FCBias);


figure

yyaxis left
plot(x,Accuracy,'-db','LineWidth',2,'MarkerIndices',i)
hAx=gca;  % avoid repetitive function calls
set(hAx,'xminorgrid','on','yminorgrid','on')

yyaxis right
semilogy(x,Mem,'-or','MarkerIndices',i);
hold on
grid

semilogy(x,Mem_16,'--^r','MarkerIndices',i);
semilogy(x,Mem_8,'-->r','MarkerIndices',i);
semilogy(x,Mem_4,'--<r','MarkerIndices',i);
semilogy(x,Mem_2,'--vr','MarkerIndices',i);

Stack = 0.75;
F = 8;

% MICROCHIP MSP430 512KB Flash, 66KB RAM
M = 66000*Stack;
yline(M,'--r')
text(180,M+10000,'MSP430','Color','r','FontSize',F)

% Silicon Labs C8051F98x 8KB Flash, 512 KB RAM
M = 512000*Stack;
yline(M,'--r')
text(1,M+100000,'C8051F98x','Color','r','FontSize',F)

% ST STM88L 64 KB Flash, 6 KB RAM
M = 6000*Stack;
yline(M,'--r')
text(180,M+1000,'STM88L','Color','r','FontSize',F)

% ST STM32L 1MB Flash, 320 KB RAM
M = 320000*Stack;
yline(M,'--r')
text(1,M+60000,'STM32L','Color','r','FontSize',F)

lgd = legend('Accuracy','Memory Footprint','INT16','INT8','INT4','INT2','Location','southoutside');
lgd.NumColumns  = 2;

% *************************************************************************