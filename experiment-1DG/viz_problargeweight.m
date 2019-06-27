%% Script to compute probability of sampling a weight exceeding a large value
%
% Pr[w(x) > y] = 1 - Pr[w(x) <= y]
%              = 1 - (Pr[w^-1(y) <= x] - Pr[-w^-1(y) <= x])
%              = 1 - (Phi[w^-1(y)] - Phi[-w^-1(y)])
%
% Author: Wouter M. Kouw
% Last updated: 27-10-2018

close all;
clearvars;

%% Visualization parameters

% Colors
cm = parula;
cmix = round(linspace(10,54,4));
clrs = cm(cmix,:);

% Legend
lgd_gamma = {};

% Line styles
lstyles = {'-', '--', '-.', ':'};

% Font size
fS = 25;

%% Probability of weight exceeding constant c, for mu_S = 0

% Source std dev
gamma = [.5 .65 .8 .95];
nG = length(gamma);

% Span of weights
N = 100;
x = linspace(-3, 3, N);
y = logspace(0, 3, N);

% Preallocate
prwc = zeros(nG, N);

for g = 1:nG
    
    if gamma(g) == 1
        prwc(g,:) = NaN(1,N);
    else
    
    % Weight function
    w = @(x,g) g.* exp(-(g.^2-1)./(2*g.^2)*x.^2);
    
    % Inverse weight functions
    iw = sqrt((2*gamma(g)^2)/(gamma(g)^2-1)*log(gamma(g)./y));
    
    % Probability of weight exceeding value y
    prwc(g,:) = 1 - (normcdf(iw, 0, gamma(g)) - normcdf(-iw, 0, gamma(g)));

    end
end

% Log-plot of weights
figure()
for g = 1:nG
    loglog(y, prwc(g,:), 'LineStyle', lstyles{g}, 'Color', 'k', 'LineWidth', 5)
    hold on
    
    lgd_gamma(g) = {['$\gamma$ = ' num2str(gamma(g), 3)]};
end

% Legend
legend(lgd_gamma, 'Interpreter', 'latex', 'Location', 'northeast');

% Axes options
xlabel('constant (c)');
ylabel('Pr[ w(x) > c ]');

% Save figure
set(gca, 'YLim', [1e-2 1e0]);
set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 500]);
saveas(gcf, 'viz/prob_wc_v01.eps', 'epsc');

%% Probability of weight exceeding constant c, for arbitrary mu_S

% Set mu_S
mu_S = -1;

% Source std dev
gamma = 0.7:0.1:0.9;
nG = length(gamma);

% Span of weights
N = 101;
x = linspace(-3, 3, N);
y = logspace(0, 3, N);

% Preallocate
prwc = zeros(nG, N);

for g = 1:nG
    
    if gamma(g) == 1
        prwc(g,:) = NaN(1,N);
    else
    
    % Weight function
    w = @(x,g,m) g.* exp(((m-x).^2./g.^2 - x.^2)/2);
    
    % Inverse weight function due to negative root
    iw_n = @(y,g,m) (m./g.^2 - sqrt(m.^2 + 2*log(y./g) - 2*g.^2.*log(y./g))./g) ./ (g.^(-2)-1);
    
    % Inverse weight function due to positive root
    iw_p = @(y,g,m) (m./g.^2 + sqrt(m.^2 + 2*log(y./g) - 2*g.^2.*log(y./g))./g) ./ (g.^(-2)-1);
    
    % Probability of weight exceeding value y
    prwc(g,:) = 1 - (normcdf(iw_p(y,gamma(g), mu_S), 0, gamma(g)) - normcdf(iw_n(y,gamma(g), mu_S), 0, gamma(g)));

    end
end

% Log-plot of weights
figure()
for g = 1:nG
    loglog(y, prwc(g,:), 'LineStyle', lstyles{g}, 'Color', 'k', 'LineWidth', 5)
    hold on
    
    lgd_gamma(g) = {['$\gamma$ = ' num2str(gamma(g), 2)]};
end

% Legend
legend(lgd_gamma, 'Interpreter', 'latex', 'Location', 'northeast');

% Axes options
xlabel('constant (c)');
ylabel('Pr[ w(x) > c ]');

% Save figure
set(gca, 'YLim', [1e-2 1e0]);
set(gca, 'FontSize', fS);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 500]);
saveas(gcf, 'viz/prob_wc_muS.eps', 'epsc');


