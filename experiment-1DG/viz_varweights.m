%% Visualize variance of importance weights
%
% Author: Wouter M. Kouw
% Last updated: 15-05-2019

close all;
clearvars;

%% Variance of weights for mu_S = 0

% Source std dev
gamma = linspace(0.7, 1, 1e6);
nG = length(gamma);

% Weight function
wx = @(x, g) g.*exp(-x^2.*(g.^2 - 1)./(2*g.^2));

% Variance function
Vw = @(g) real( gamma.^2 ./ sqrt(2*gamma.^2 - 1) - 1 );

% Log-plot of weights
figure()

% Logplot
plot(gamma, Vw(gamma), 'LineWidth', 5, 'Color', 'k')

% Axes options
xlabel('Source standard deviation ($$\gamma$$)', 'Interpreter', 'latex');
ylabel('$$V_{\cal S}[w(x)]$$', 'Interpreter', 'latex');

set(gca, 'XScale', 'lin', 'XLim', [gamma(1) gamma(end)]);
set(gca, 'YScale', 'log', 'YLim', [1e-2 1e3]);

% Save figure
set(gca, 'FontSize', 25);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 400]);

saveas(gcf, 'viz/Variance_weights_muS0.eps', 'epsc');
saveas(gcf, 'viz/Variance_weights_muS0.png');

%% Variance of weights for mu_S = -1

% Source std dev
gamma = linspace(0.7, 1, 1e6);
nG = length(gamma);

% Weight function
wx = @(x, g) g.*exp(-1/2*(x^2 + g^(-2)*(x+1)^2));

% Variance function
Vw = @(g,m) exp(m^2 ./ (2*g.^2 - 1)).*g.^2 ./(2*g.^2 -1) -1;

% Log-plot of weights
figure()

% Logplot
plot(gamma, Vw(gamma,-1), 'LineWidth', 5, 'Color', 'k')

% Axes options
xlabel('Source standard deviation ($$\gamma$$)', 'Interpreter', 'latex');
ylabel('$$V_{\cal S}[w(x)]$$', 'Interpreter', 'latex');

set(gca, 'XScale', 'lin', 'XLim', [gamma(1) gamma(end)]);
set(gca, 'YScale', 'log', 'YLim', [1e0 1e4]);

% Save figure
set(gca, 'FontSize', 25);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 400]);

saveas(gcf, 'viz/Variance_weights_muS-1.eps', 'epsc');
saveas(gcf, 'viz/Variance_weights_muS-1.png');

