% Plot results from 2D Gaussian experiment
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

di = 1; while exist([savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di-1) '.mat'];
disp(['Done. Writing to ' fn]);
load(fn, 'delta', 'Lambda', 'W', 'R_S', 'R_W', 'R_C', 'R_T', ...
    'lambda_S', 'lambda_W', 'lambda_C', 'lambda_T', ...
    'Rh_T', 'Rh_S', 'Rh_W', 'Rh_C', 'Rh_T');

%% Experimental parameter

% Number of folds
nF = 5;

% Number of repeats
nR = 1e4;

% Sample sizes
N = 50;   % source data
M = 1000;  % target training

% Importance weight estimator
iwT = 'true';

% Lambda range
Lambda = logspace(-6,6,201);
nL = length(Lambda);

% Source parameters
mu_S = [-1 0];
Sigma_S = [1 0; 0 1];

% Target parameters
mu_T = [0 0];
Sigma_T = eye(2);

% Target parameter changes
rho = 1./2;
shifttype = 'var';
delta = 2.^[-1.5:.25:0];
nD = length(delta);

% Plot settings
mS = 100;
fS = 25;
lW = 4;

% Colors
cm = parula;
cmix = round(linspace(10,54,4));
clrs = cm(cmix,:);

%% Limit analysis to problematic cases

mRh_S = zeros(nD, nL);
mRh_W = zeros(nD, nL);
mRh_C = zeros(nD, nL);
mRh_T = zeros(nD, nL);

vRh_S = zeros(nD, nL);
vRh_W = zeros(nD, nL);
vRh_C = zeros(nD, nL);
vRh_T = zeros(nD, nL);

mR_S = zeros(nD, 1);
mR_W = zeros(nD, 1);
mR_C = zeros(nD, 1);
mR_T = zeros(nD, 1);

vR_S = zeros(nD, 1);
vR_W = zeros(nD, 1);
vR_C = zeros(nD, 1);
vR_T = zeros(nD, 1);

mlS = zeros(nD, 1);
mlW = zeros(nD, 1);
mlC = zeros(nD, 1);
mlT = zeros(nD, 1);

for d = 1:nD
    
    % Compute variance of weights across repetitions
    varW = squeeze(nanvar(W(d,:,:), [], 2));
    
    % Cut-off constant
    cutoff = 10;
    limix = (varW > cutoff);
    disp(['Mean number of selected ', num2str(mean(limix))])
    
    mRh_S(d, :) = nanmean(Rh_S(d, :, limix), 3);
    mRh_W(d, :) = nanmean(Rh_W(d, :, limix), 3);
    mRh_C(d, :) = nanmean(Rh_C(d, :, limix), 3);
    mRh_T(d, :) = nanmean(Rh_T(d, :, limix), 3);
    
    vRh_S(d, :) = nanvar(Rh_S(d, :, limix), [], 3);
    vRh_W(d, :) = nanvar(Rh_W(d, :, limix), [], 3);
    vRh_C(d, :) = nanvar(Rh_C(d, :, limix), [], 3);
    vRh_T(d, :) = nanvar(Rh_T(d, :, limix), [], 3);
    
    mR_S(d) = nanmean(R_S(d, limix), 2);
    mR_W(d) = nanmean(R_W(d, limix), 2);
    mR_C(d) = nanmean(R_C(d, limix), 2);
    mR_T(d) = nanmean(R_T(d, limix), 2);
    
    vR_S(d) = nanvar(R_S(d, limix), [], 2);
    vR_W(d) = nanvar(R_W(d, limix), [], 2);
    vR_C(d) = nanvar(R_C(d, limix), [], 2);
    vR_T(d) = nanvar(R_T(d, limix), [], 2);
    
    mlS(d) = nanmean(Lambda(lambda_S(d, limix)), 2);
    mlW(d) = nanmean(Lambda(lambda_W(d, limix)), 2);
    mlC(d) = nanmean(Lambda(lambda_C(d, limix)), 2);
    mlT(d) = nanmean(Lambda(lambda_T(d, limix)), 2);
end

%% Plot estimated regularization parameters

% Initialize figure
figure()
hold on

% Plot mean lambda
plot(delta, mlW, 'Color', clrs(2,:), 'LineWidth', lW)
plot(delta, mlC, 'Color', clrs(3,:), 'LineWidth', lW)
plot(delta, mlT, 'Color', clrs(4,:), 'LineWidth', lW)

% Set axes properties
xlabel('$\gamma$', 'interpreter', 'latex')
ylabel('Mean estimate ($\hat{\lambda}$)', 'interpreter', 'latex')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log', 'YLim', [10^-3, 10^6]);
title(['Mean target risk for selected lambdas, by source variance' newline 'for sets with V[w] > ' num2str(cutoff)], 'FontSize', fS-5);
legend({'$\hat{R}_{\cal W}$', '$\hat{R}_{\cal \beta}$', '$\hat{R}_{\cal T}$'}, 'Interpreter', 'latex', 'Location', 'SouthEast', 'FontSize', fS+5)
set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 600])

% Write figure to file
if save_figs
    saveas(gcf, ['viz/delta_mla_varW' num2str(cutoff) '_nR' num2str(nR) '.eps'], 'epsc')
end

%% Plot mean target risks

% Initalize figure
figure()
hold on

% Plot mean empirical risks
plot(delta, mR_W, 'Color', clrs(2,:), 'LineWidth', lW)
plot(delta, mR_C, 'Color', clrs(3,:), 'LineWidth', lW)
plot(delta, mR_T, 'Color', clrs(4,:), 'LineWidth', lW)

% Set axes properties
xlabel('$\gamma$', 'interpreter', 'latex')
ylabel('Mean target risk ($R$)', 'interpreter', 'latex')
set(gca, 'XScale', 'log');
set(gca, 'YLim', [0.65 0.9]);
title(['Mean target risk for selected lambdas, by source variance' newline 'for sets with V[w] > ' num2str(cutoff)], 'FontSize', fS-5);
legend({'$\hat{R}_{\cal W}$', '$\hat{R}_{\cal \beta}$', '$\hat{R}_{\cal T}$'}, 'Interpreter', 'latex', 'Location', 'NorthEast', 'FontSize', fS+5)
set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 600])

% Write figure to file
if save_figs
    saveas(gcf, ['viz/delta_mR_varW' num2str(cutoff) '_nR' num2str(nR) '.eps'], 'epsc')
end



