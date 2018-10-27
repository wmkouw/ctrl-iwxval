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

%% Inverse weight function

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
    w = gamma(g).*exp(-(gamma(g).^2-1)./(2*gamma(g).^2)*x.^2);
    
    % Inverse weight functions
    iw = sqrt((2*gamma(g)^2)/(gamma(g)^2-1)*log(gamma(g)./y));
    
    % Derivative of inverse weight function
    diw = 1./(2*iw).*(-2*gamma(g).^2)./((gamma(g)^2-1)*y);
    
    % Probability of weight exceeding value y
    prwc(g,:) = 1 - (normcdf(iw, 0, gamma(g)) - normcdf(-iw, 0, gamma(g)));

    end
end

%% Plot results

% Colors
cm = parula;
cmix = round(linspace(10,54,4));
clrs = cm(cmix,:);

% Legend
lgd_gamma = {};

% Log-plot of weights
figure()
for g = 1:nG
    loglog(y, prwc(g,:), 'Color', clrs(g,:),'LineWidth', 5)
    hold on
    
    lgd_gamma(g) = {['$\gamma$ = ' num2str(gamma(g), 3)]};
end

% Legend
legend(lgd_gamma, 'Interpreter', 'latex', 'Location', 'SouthWest');

% Axes options
title(['Cumulative probability of importance weight' newline 'exceeding value c (for source std.dev. = ', num2str(gamma), ')']);
xlabel('constant (c)');
ylabel('Pr[ w(x) > c ]');

% Save figure
set(gca, 'FontSize', 25);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 600]);
saveas(gcf, 'prob_wc_v01.eps', 'epsc');


