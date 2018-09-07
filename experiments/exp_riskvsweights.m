close all;
clearvars;

% Import utility functions
addpath(genpath('../util'))

% Make results and visualization directories
mkdir('viz');

%% Experimental parameters

% Number of repetitions
nR = 1e5    ;

% Whether to save figures
sav = true;

%% Visualization parameters

% Font size
fS = 20;

% Marker size
mS = 20;

% Line width
lW = 4;

%% Problem setting

% Sample sizes
N = 10;
M = 10;

% Source parameters
mu_S = 0;
si_S = sqrt(2)./2;

% Target parameters
mu_T = 0;
si_T = 1;

% 2D grid
nU = 501;
ul = [-20 +20];
u1 = linspace(ul(1),ul(2),nU);

%% Hypothesis and risk parameters

% Fixed parameter theta
th = 0.2;

% Analytical solution to target risk
RT = @(theta) theta.^2 - 2*theta./sqrt(pi) + 1;

%% Distribution functions

% Priors
pY = @(y) 1./2;

% Class-posteriors (pYS == pYT for covariate shift)
pYS = @(y,x) normcdf(y*x);
pYT = @(y,x) pYS(y,x);

% Source marginal distribution
pS = @(x) normpdf(x, mu_S, si_S);

% Source class-conditional distributions
pSY = @(y,x) (pYS(y,x) .* pS(x))./pY(y);

% Target marginal distribution
pT = @(x) normpdf(x, mu_T, si_T);

% Target class-conditional distributions
pTY = @(y,x) (pYT(y,x) .* pT(x))./pY(y);

% Importance weights
IW = @(x) pT(x) ./ pS(x);

% Helper functions for rejection sampling
pS_yn = @(x) pSY(-1,x);
pS_yp = @(x) pSY(+1,x);
pT_yn = @(x) pTY(-1,x);
pT_yp = @(x) pTY(+1,x);

% Preallocate
W = zeros(N, nR);
Rh_S = zeros(nR,1);
Rh_W = zeros(nR,1);
Rh_T = zeros(nR,1);
Rh_C = zeros(nR,1);
beta = zeros(nR,1);

for r = 1:nR
    
    % Report progress over repetitions
    if (rem(r,nR./10)==1)
        fprintf('At repetition \t%i/%i\n', r, nR)
    end
    
    % Rejection sampling of target validation data
    const = 1.2./sqrt(2*pi*si_T);
    ss_Tn = round(M.*pY(-1));
    ss_Tp = M - ss_Tn;
    Zy_n = sampleDist1D(pT_yn, const, ss_Tn, ul);
    Zy_p = sampleDist1D(pT_yp, const, ss_Tp, ul);
    
    % Rejection sampling of source data
    const = 1.2./sqrt(2*pi*si_S);
    ss_Sn = round(N.*pY(-1));
    ss_Sp = N - ss_Sn;
    Xy_n = sampleDist1D(pS_yn, const, ss_Sn, ul);
    Xy_p = sampleDist1D(pS_yp, const, ss_Sp, ul);
    
    % Concatenate to datasets
    Z = [Zy_n; Zy_p];
    X = [Xy_n; Xy_p];
    u = [-ones(size(Zy_n,1),1); ones(size(Zy_p,1),1)];
    y = [-ones(size(Xy_n,1),1); ones(size(Xy_p,1),1)];
    
    % Importance weights
    W(:, r) = IW(X);
    
    % Target risk of estimated theta
    Rh_S(r) = mean((X*th - y).^2, 1);
    Rh_W(r) = mean((X*th - y).^2 .* W(:, r), 1);
    Rh_T(r) = mean((Z*th - u).^2, 1);
    
    % Estimate beta coefficient
    weighted_loss = (X*th - y).^2.*W(:, r);
    a_i = (weighted_loss - mean(weighted_loss, 1));
    b_i = (W(:, r) - mean(W(:, r), 1));
    beta(r) = sum(a_i .* b_i, 1) ./ sum(b_i.^2,1);
    
    % Compute controlled estimate
    Rh_C(r) = mean((X*th - y).^2 .* W(:, r) - beta(r)*(W(:,r) - 1), 1);
    
end

%% Compute properties of weight sets

maxW = max(W, [], 1)';
varW = var(W, [], 1)';

%% Plot maximum weight versus risk

figure()

% Maximum weight versus weighted risk
plot(linspace(0, max(maxW(:)), 2), [RT(th), RT(th)], 'k', 'LineWidth', lW);
hold on
s = scatter(maxW, Rh_W, 'k', 'filled');
alpha(s, 0.2);

% Axes information
xlabel('$$\max[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{W}$$', 'Interpreter', 'latex')
set(gca, 'FontSize', fS);

% Set figure information
title('Maximum weight vs uncontrolled estimate');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/maxW_vs_RhW_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png']);
end

%% Plot maximum weight versus risk

figure()

% Maximum weights versus controlled estimate
plot(linspace(0, max(maxW(:)), 2), [RT(th), RT(th)], 'k', 'LineWidth', lW);
hold on
s = scatter(maxW, Rh_C, 'b', 'filled');
alpha(s, 0.2);

% Axes information
xlabel('$$\max[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{C}$$', 'Interpreter', 'latex')
set(gca, 'FontSize', fS);

% Set figure information
title('Maximum weight vs controlled estimate');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/maxW_vs_RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png']);
end

%% Plot maximum weight versus absolute deviation in risk

figure()

% Maximum weights versus difference in risks
s = scatter(maxW, Rh_C - Rh_W, 'r', 'filled');
alpha(s, 0.2);

% Axes information
xlabel('$$\max[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{C} - \hat{R}_{W}$$', 'Interpreter', 'latex')
set(gca, 'FontSize', fS);

% Set figure information
title('Maximum weight vs risk difference');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/maxW_vs_RhW-RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png']);
end

%% Plot weight variance versus risk

figure()

% Variance of weights versus weighted estimates
plot(linspace(0, max(maxW(:)), 2), [RT(th), RT(th)], 'k', 'LineWidth', lW);
hold on
s = scatter(varW, Rh_W, 'k', 'filled');
alpha(s, 0.2)

% Axes information
xlabel('var$$[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_W$$', 'Interpreter', 'latex')
set(gca, 'XScale', 'log', 'FontSize', fS);

% Set figure information
title('Weight variance vs uncontrolled estimate');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/varW_vs_RhW_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

%% Plot weight variance versus risk

figure()

% Variance of weights versus controlled estimate
plot(linspace(0, max(maxW(:)), 2), [RT(th), RT(th)], 'k', 'LineWidth', lW);
hold on
s = scatter(varW, Rh_C, 'b', 'filled');
alpha(s, 0.2)

% Axes information
xlabel('var$$[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_C$$', 'Interpreter', 'latex')
set(gca, 'XScale', 'log', 'FontSize', fS);

% Set figure information
title('Weight variance vs controlled estimate');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/varW_vs_RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

%% Plot weight variance versus risk difference

figure()

% Variance of weights versus difference in risks
s = scatter(varW, Rh_C - Rh_W, 'r', 'filled');
alpha(s, 0.2)

% Axes information
xlabel('var$$[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_C - \hat{R}_W$$', 'Interpreter', 'latex')
set(gca, 'XScale', 'log', 'FontSize', fS);

% Set figure information
title('Weight variance vs risk difference');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/varW_vs_RhW-RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

