close all;
clearvars;

% Import utility functions
addpath(genpath('../util'))

% Make results and visualization directories
mkdir('results');
mkdir('viz');

% Whether to save figures
sav = true;

% Visualization parameters
fS = 25;
lW = 4;
xx = linspace(-5,5,1001);

%% Problem setting

% Source parameters
mu_S = -1;
si_S = 0.75;

% Target parameters
mu_T = 0;
si_T = 1;

% 2D grid
nU = 501;
ul = [-10 +10];
u1 = linspace(ul(1),ul(2),nU);

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

%% Data distributions


% Initialize figure for both domains
fg10 = figure(10);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pS(xx), 'k', 'LineWidth', lW)
plot(xx, pT(xx), 'k', 'LineStyle', ':', 'LineWidth', lW)

% Axes information
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p(x)$$', 'Interpreter', 'latex');
legend({'$$p_{\cal S}$$', '$$p_{\cal T}$$'}, 'Interpreter', 'latex', 'FontSize', fS+5);
set(gca, 'XLim', [-3 3], 'YLim', [0 0.6], 'FontSize', fS);
set(fg10, 'Color', 'w', 'Position', [0 0 1200 400]);

if sav
    saveas(fg10, ['viz/1Dsetting_datadists_' num2str(si_T, 3) '.eps'], 'epsc');
end

% Initialize figure for source domain
fg1 = figure(1);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pS(xx), 'k', 'LineWidth', lW)

% Axes information
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal S}(x)$$', 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg1, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg1, ['viz/pSx_siS' num2str(si_S, 3) '.eps'], 'epsc');
end

% Initialize figure for source domain
fg2 = figure(2);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pT(xx), 'k', 'LineWidth', lW)

% Axes information
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal T}(x)$$', 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg2, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg2, ['viz/pTx_siT' num2str(si_T, 3) '.eps'], 'epsc');
end

%% Class-conditional distributions

% Initialize figure for source domain
fg3 = figure(3);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pS_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', ' -1')
plot(xx, pS_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal S}(x|y)$$', 'Interpreter', 'latex');
% title(['Source domain, $$\sigma_S$$ = ' num2str(si_S)], 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg3, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg3, ['viz/pSxy_siS' num2str(si_S, 3) '.eps'], 'epsc');
end

% Initialize figure for target domain
fg4 = figure(4);
hold on

% Plot target class-conditional distributions, p_S(x|y)
plot(xx, pT_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', ' -1')
plot(xx, pT_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal T}(x|y)$$', 'Interpreter', 'latex');
% title(['Target domain, $$\sigma_T$$ = ' num2str(si_T)], 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg4, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg4, ['viz/pTxy_siT' num2str(si_T, 3) '.eps'], 'epsc');
end

%% Class-conditional distributions

% Initialize figure for source domain
fg5 = figure(5);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pYS(-1, xx), 'r', 'LineWidth', lW, 'DisplayName', ' -1')
plot(xx, pYS(+1, xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal S}(y|x)$$', 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg5, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg5, ['viz/pSyx_siS' num2str(si_S, 3) '.eps'], 'epsc');
end

% Initialize figure for target domain
fg6 = figure(6);
hold on

% Plot target class-conditional distributions, p_S(x|y)
plot(xx, pYT(-1, xx), 'r', 'LineWidth', lW, 'DisplayName', ' -1')
plot(xx, pYT(+1, xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal T}(y|x)$$', 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg6, 'Color', 'w', 'Position', [0 0 1200 800]);

if sav
    saveas(fg6, ['viz/pTyx_siT' num2str(si_T, 3) '.eps'], 'epsc');
end
