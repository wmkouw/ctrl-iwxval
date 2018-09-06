%% Weight distribution computations

close all;
clearvars;

%% Inverse weight function

% Source std dev
gamma = sqrt(2)./2;

% Span of random varirable
x = linspace(-3,3,100);

% Span of weights
y = logspace(0,2,100);

% Weight function
w = gamma.*exp(-(gamma.^2-1)./(2*gamma.^2)*x.^2);

% Inverse weight function
iw = sqrt((2*gamma^2)/(gamma^2-1)*log(gamma./y));

% Derivative of inverse weight function
diw = 1./(2*iw).*(-2*gamma.^2)./((gamma^2-1)*y);

%% Probability of weight exceeding some value
%
% Pr[w(x) > y] = 1 - Pr[w(x) <= y] 
%              = 1 - (Pr[w^-1(y) <= x] - Pr[-w^-1(y) <= x])
%              = 1 - (Phi[w^-1(y)] - Phi[-w^-1(y)])

% Cumulative normal
Pr = 1 - (normcdf(iw, 0, gamma) - normcdf(-iw, 0, gamma));

%% Plot results

% Log-plot
loglog(y,Pr)

% Axes options
title(['Cumulative probability of importance weight' newline 'exceeding value c (for source std.dev. = ', num2str(gamma), ')']);
xlabel('constant (c)');
ylabel('Pr[w(x) > c]');

% Save figure
set(gcf, 'Color', 'w');
saveas(gcf, 'prob_wc_v01.png');


