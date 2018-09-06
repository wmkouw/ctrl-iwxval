close all;
clearvars;

% Import utility functions
addpath(genpath('../util'))

% Make results and visualization directories
mkdir('viz');

%% Experimental parameters

% Number of repetitions
nR = 1e5;

% Whether to save figures
sav = true;

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
Rh_CW = zeros(nR,1);
Rh_C1 = zeros(nR,1);
betaW = zeros(nR,1);
beta1 = zeros(nR,1);

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
    betaW(r) = mean(((X*th - y).^2.*W(:, r) - mean((X*th - y).^2 .* W(:, r), 1)) .* (W(:, r) - mean(W(:, r), 1)), 1) ./ mean((W(:, r) - mean(W(:, r), 1)).^2,1);
    beta1(r) = mean(((X*th - y).^2.*W(:, r) - mean((X*th - y).^2 .* W(:, r), 1)) .* (W(:, r) - 1), 1) ./ mean((W(:, r) - 1).^2,1);
    Rh_CW(r) = mean((X*th - y).^2 .* W(:, r) - betaW(r)*(W(:,r) - 1), 1);
    Rh_C1(r) = mean((X*th - y).^2 .* W(:, r) - beta1(r)*(W(:,r) - 1), 1);
    
end

%% Plot maximum weight versus risk

% Visualization parameters
fS = 24;
mS = 20;
lW = 5;

% Select large weights
maxW = max(W, [], 1)';

figure()
hold on
% Plot normalized histograms
% semilogx(maxW, Rh_W, 'k.', 'MarkerSize', mS);
plot(maxW, Rh_W, 'k.', 'MarkerSize', mS);
hold on
xlim = get(gca, 'XLim');
plot(linspace(xlim(1), xlim(2), 2), [RT(th), RT(th)], 'r', 'LineWidth', lW);

lines = findobj(gcf, 'Type', 'Line');
legend([lines(end), lines(1)], {'without control', 'true target risk'})

% Axes information
xlabel('$$\max[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{W}$$', 'Interpreter', 'latex')
% set(gca, 'YLim', [0 30], 'XLim', [0 1000], 'FontSize', fS);
set(gca, 'FontSize', fS);

% Set figure information
title('Maximum weight versus risk');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/maxW_vs_RhW_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png']);
end


%% Plot maximum weight versus risk

% Visualization parameters
fS = 24;
mS = 20;
lW = 5;

% Select large weights
maxW = max(W, [], 1)';

figure()
hold on
% Plot normalized histograms
% semilogx(maxW, Rh_W, 'k.', 'MarkerSize', mS);
plot(maxW, Rh_CW, 'b.', 'MarkerSize', mS);
hold on
xlim = get(gca, 'XLim');
plot(linspace(xlim(1), xlim(2), 2), [RT(th), RT(th)], 'r', 'LineWidth', lW);

lines = findobj(gcf, 'Type', 'Line');
legend([lines(end), lines(1)], {'with control', 'true target risk'})

% Axes information
xlabel('$$\max[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{C}$$', 'Interpreter', 'latex')
% set(gca, 'YLim', [0 30], 'XLim', [0 1000], 'FontSize', fS);
set(gca, 'FontSize', fS);

% Set figure information
title('Maximum weight versus risk');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/maxW_vs_RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png']);
end

%% Plot weight variance versus risk

% Visualization parameters
fS = 24;
mS = 20;
lW = 5;

% Select large weights
varW = var(W, [], 1)';

figure()
% Plot normalized histograms
semilogx(varW, Rh_W, 'k.', 'MarkerSize', mS);
% plot(varW, Rh_W, 'k.', 'MarkerSize', mS);
hold on
xlim = get(gca, 'XLim');
plot(linspace(xlim(1), xlim(2), 2), [RT(th), RT(th)], 'r', 'LineWidth', lW);

lines = findobj(gcf, 'Type', 'Line');
legend([lines(end), lines(1)], {'without control', 'true target risk'})

% Axes information
xlabel('var$$[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_W$$', 'Interpreter', 'latex')
% set(gca, 'YLim', [0 30], 'XLim', [10^-3 10^5], 'FontSize', fS);
set(gca, 'FontSize', fS);

% Set figure information
title('Variance of weights versus risk');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/varW_vs_RhW_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

%% Plot weight variance versus risk

% Visualization parameters
fS = 24;
mS = 20;
lW = 5;

% Select large weights
varW = var(W, [], 1)';

figure()
% Plot normalized histograms
semilogx(varW, Rh_CW, 'b.', 'MarkerSize', mS);
% plot(varW, Rh_W, 'k.', 'MarkerSize', mS);
hold on
xlim = get(gca, 'XLim');
plot(linspace(xlim(1), xlim(2), 2), [RT(th), RT(th)], 'r', 'LineWidth', lW);

lines = findobj(gcf, 'Type', 'Line');
legend([lines(end), lines(1)], {'with control', 'true target risk'})

% Axes information
xlabel('var$$[w]$$', 'Interpreter', 'latex')
ylabel('$$\hat{R}_{C}$$', 'Interpreter', 'latex')
% set(gca, 'YLim', [0 30], 'XLim', [10^-3 10^5], 'FontSize', fS);
set(gca, 'FontSize', fS);

% Set figure information
title('Variance of weights versus risk');
set(gcf, 'Color', 'w', 'Position', [10 100 1000 1000]);

if sav
    saveas(gcf, ['viz/varW_vs_RhC_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

