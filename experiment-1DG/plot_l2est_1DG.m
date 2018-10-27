% Plot results from 1D Gaussian experiment
%
% Author: Wouter M. Kouw
% Last updated: 27-10-2018

close all;
clearvars;

save_figs = true;
san = false;
viz = false;

if ~exist('viz', 'dir'); mkdir('viz'); end

%% Load data

savnm = 'results/';
iwT = 'true';

di = 1; while exist([savnm '1DG_split_iw-' iwT '_varshift_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm '1DG_split_iw-' iwT '_varshift_' num2str(di-1) '.mat'];
disp(['Loading ' fn]);
load(fn);

%% Parameters

% Subselection index
nD = length(delta);
ix = 1:nD-2;

% Plot parameters
lW = 5;
fS = 20;
he = 600;

% Colors
cm = parula;
cmix = round(linspace(10,54,4));

%% Plot empirical risks

fg1 = figure(1);
hold on

% Plot risks
plot(delta(ix), mean(RTh.V(ix,:),2), '-', 'Color', cm(cmix(1),:), 'LineWidth', lW, 'DisplayName','$S$');
plot(delta(ix), mean(RTh.W(ix,:),2), '-', 'Color', cm(cmix(2),:), 'LineWidth', lW, 'DisplayName','$W$');
plot(delta(ix), mean(RTh.B(ix,:),2), '-', 'Color', cm(cmix(3),:), 'LineWidth', lW, 'DisplayName','$\beta$');
plot(delta(ix), mean(RTh.Z(ix,:),2), '-', 'Color', cm(cmix(4),:), 'LineWidth', lW, 'DisplayName','$T$');

% Set axes properties
set(gca, 'XScale', 'log')
set(gca, 'FontSize', fS);
xlabel('$\delta$', 'Interpreter', 'LaTex', 'FontSize', fS);
ylabel('$$\hat{R}$$', 'Interpreter', 'LaTex', 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1600 600]);
set(gcf, 'Color', 'w', 'Position', [0 0 1600 600]);
legend({'$${\cal S}$$', '$${\cal W}$$', '$$\beta$$', '$${\cal T}$$'}, 'Location','northwest', 'Interpreter', 'latex');

% Write figure to file
if sav
    saveas(gcf, ['viz/1DG_Rh_N' num2str(N) '_nF' num2str(nF) '_nR' num2str(nR) '.eps'], 'epsc'); 
end

%% Plot difference in risks

% Initialize figure
fg3 = figure(3);
hold on

% Plot risks
plot(delta(ix), mean(RTh.V(ix,:) - RTh.Z(ix,:),2), '-', 'Color', cm(cmix(1),:), 'LineWidth', lW);
plot(delta(ix), mean(RTh.W(ix,:) - RTh.Z(ix,:),2), '-', 'Color', cm(cmix(2),:), 'LineWidth', lW);
plot(delta(ix), mean(RTh.B(ix,:) - RTh.Z(ix,:),2), '-', 'Color', cm(cmix(3),:), 'LineWidth', lW);

% Set axes properties
set(gca, 'FontSize', fS);
xlabel('$\delta$', 'Interpreter', 'LaTex', 'FontSize', fS);
ylabel('$$\hat{R}$$', 'Interpreter', 'LaTex', 'FontSize', fS);
set(gca, 'XScale', 'log')
set(gcf, 'Color', 'w', 'Position', [100 100 1600 600]);
set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [0 0 1600 600]);
legend({'$${\cal S}$$', '$${\cal W}$$', '$$\beta$$'}, 'Location','northwest', 'Interpreter', 'latex');

% Write figure to file
if sav
    saveas(gcf, ['viz/1DG_RhRZ_N' num2str(N) '_nF' num2str(nF) '_nR' num2str(nR) '.eps'], 'epsc'); 
end
