close all;
clearvars;

% Import utility functions
addpath(genpath('../util'))

% Make results and visualization directories
mkdir('results');
mkdir('viz');

% Whether to save figures
sav = true;

%% Problem setting

% Source parameters
mu_S = 0;
si_S = .5;

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

%% Visualize setting


% Visualization parameters
fS = 30;
lW = 5;
xx = linspace(-5,5,1001);

% Initialize figure for source domain
fg1 = figure(1);
hold on

% Plot source class-conditional distributions, p_S(x|y)
plot(xx, pS_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', '-1')
plot(xx, pS_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal S}(x|y)$$', 'Interpreter', 'latex');
title(['Source domain, $$\sigma_S$$ = ' num2str(si_S)], 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg1, 'Color', 'w', 'Position', [0 0 1200 600]);

if sav
    saveas(fg1, ['viz/source_siS' num2str(si_S) '.png']);
end

% Initialize figure for target domain
fg2 = figure(2);
hold on

% Plot target class-conditional distributions, p_S(x|y)
plot(xx, pT_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', '-1')
plot(xx, pT_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');

% Axes information
legend('show')
xlabel('$$x$$', 'Interpreter', 'latex');
ylabel('$$p_{\cal T}(x|y)$$', 'Interpreter', 'latex');
title(['Target domain, $$\sigma_T$$ = ' num2str(si_T)], 'Interpreter', 'latex');
set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
set(fg2, 'Color', 'w', 'Position', [0 0 1200 600]);

if sav
    saveas(fg2, ['viz/target_siT' num2str(si_T) '.png']);
end
