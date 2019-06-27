% Plot results from 2D Gaussian experiment
%
% Author: Wouter M. Kouw
% Last updated: 27-10-2018

close all;
clearvars;

save_figs = true;
san = false;
viz = false;

if ~exist('viz', 'dir')
    mkdir('viz'); 
end

%% Load data

savnm = 'results/';
iwT = 'gauss';

di = 1;
while exist([savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'], 'file')
    di = di+1; 
end
di = 6;
fn = [savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di-1) '.mat'];
disp(['Loading ' fn]);
load(fn);

%% Experimental parameter

nG = length(gamma);

% Plot settings
mS = 100;
fS = 25;
lW = 5;

% Colors
cm = parula;
cmix = round(linspace(10,54, 4));
clrs = cm(cmix,:);

%% Limit analysis to problematic cases

% Variance cutoff
cutoff = 90;

p = zeros(nG, 1);

mRS_lim = zeros(nG, 1);
mRW_lim = zeros(nG, 1);
mRC_lim = zeros(nG, 1);
mRM_lim = zeros(nG, 1);
mRT_lim = zeros(nG, 1);

semRS_lim = zeros(nG, 1);
semRW_lim = zeros(nG, 1);
semRC_lim = zeros(nG, 1);
semRM_lim = zeros(nG, 1);
semRT_lim = zeros(nG, 1);

mRS_all = zeros(nG, 1);
mRW_all = zeros(nG, 1);
mRC_all = zeros(nG, 1);
mRM_all = zeros(nG, 1);
mRT_all = zeros(nG, 1);

semRS_all = zeros(nG, 1);
semRW_all = zeros(nG, 1);
semRC_all = zeros(nG, 1);
semRM_all = zeros(nG, 1);
semRT_all = zeros(nG, 1);


for g = 1:nG
    
    % Cut-off constant
    limix = (Vw(g,:) >= prctile(Vw(g,:), cutoff));
    disp(['Mean number of selected ', num2str(mean(limix))])
    
    % Perform significance tests
    for c = 1:nG
        p(c) = signrank(R_W(c,limix)', R_C(c,limix)');
    end
    disp(['p-value between R_W and R_C for gamma ' num2str(gamma(g)) ' = ' num2str(p(c))]);
    
    % Compute mean selected lambda's
    mRS_lim(g) = mean(R_S(g, limix), 2);
    mRW_lim(g) = mean(R_W(g, limix), 2);
    mRC_lim(g) = mean(R_C(g, limix), 2);
    mRM_lim(g) = mean(R_M(g, limix), 2);
    mRT_lim(g) = mean(R_T(g, limix), 2);
    
    % Compute standard errors of the means
    semRS_lim(g) = std(R_S(g, limix), [], 2)./nR;
    semRW_lim(g) = std(R_W(g, limix), [], 2)./nR;
    semRC_lim(g) = std(R_C(g, limix), [], 2)./nR;
    semRM_lim(g) = std(R_M(g, limix), [], 2)./nR;
    semRT_lim(g) = std(R_T(g, limix), [], 2)./nR;
    
    % Compute mean selected lambda's
    mRS_all(g) = mean(R_S(g, :), 2);
    mRW_all(g) = mean(R_W(g, :), 2);
    mRC_all(g) = mean(R_C(g, :), 2);
    mRM_all(g) = mean(R_M(g, :), 2);
    mRT_all(g) = mean(R_T(g, :), 2);
    
    % Compute standard errors of the means
    semRS_all(g) = std(R_S(g, :), [], 2)./nR;
    semRW_all(g) = std(R_W(g, :), [], 2)./nR;
    semRC_all(g) = std(R_C(g, :), [], 2)./nR;
    semRM_all(g) = std(R_M(g, :), [], 2)./nR;
    semRT_all(g) = std(R_T(g, :), [], 2)./nR;
    
end

%% Plot

figure()
hold on

errorbar(gamma(gix), mRW_lim(gix), semRW_lim(gix), '--', 'Color', clrs(2,:), 'LineWidth', lW)
errorbar(gamma(gix)*1.01, mRC_lim(gix), semRC_lim(gix), '--', 'Color', clrs(3,:), 'LineWidth', lW)

errorbar(gamma(gix), mRW_all(gix), semRW_all(gix), '-', 'Color', clrs(2,:), 'LineWidth', lW)
errorbar(gamma(gix)*1.01, mRC_all(gix), semRC_all(gix), '-', 'Color', clrs(3,:), 'LineWidth', lW)
errorbar(gamma(gix)*1.02, mRT_all(gix), semRT_all(gix), '-', 'Color', clrs(4,:), 'LineWidth', lW)

gamma_ = gamma(gix);
set(gca, 'XScale', 'lin', 'XLim', [gamma_(1), gamma_(end)]);

% Set axes properties
xlabel('Source variance ($\gamma$)', 'interpreter', 'latex')
ylabel('Mean target risk ($\bar{R}_{\cal T}$)', 'interpreter', 'latex')
legend({'$\hat{R}_{\hat{w}}>$', ...
        '$\hat{R}_{\hat{\beta}}>$', ...
        '$\hat{R}_{\hat{w}}$', ...
        '$\hat{R}_{\hat{\beta}}$', ...
        '$\hat{R}_{\cal T}$'}, ...
       'Interpreter', 'latex', ...
       'Location', 'NorthEastOutside', ...
       'FontSize', fS)

set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 500])

% Write figure to file
if save_figs
    saveas(gcf, ['viz/2DG_iwe-' iwT '_Vwcut' num2str(cutoff) '_nR' num2str(nR) '.eps'], 'epsc')
    saveas(gcf, ['viz/2DG_iwe-' iwT '_Vwcut' num2str(cutoff) '_nR' num2str(nR) '.png'])
end


