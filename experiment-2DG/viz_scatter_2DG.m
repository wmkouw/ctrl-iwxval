function viz_scatter_2DG(Zn, Zp, Xn, Xp, varargin)
% Contour plot the class-conditional distributions

% Parse arguments
p = inputParser;
addOptional(p, 'mS', 120);
addOptional(p, 'fS', 10);
addOptional(p, 'u1l', [-6 +6]);
addOptional(p, 'u2l', [-6 +6]);
addOptional(p, 'mu_S', [0 0]);
addOptional(p, 'mu_T', [0 0]);
addOptional(p, 'Sigma_S', eye(2));
addOptional(p, 'Sigma_T', eye(2));
addOptional(p, 'marg', false);
addOptional(p, 'savnmS', '');
addOptional(p, 'savnmT', '');
parse(p, varargin{:});

% Initialize figure for source data
figure()
hold on

% Scatter source data
scatter(Xn(:,1),Xn(:,2), p.Results.mS, 'filled', 'rs', 'DisplayName', 'x|y=-1');
scatter(Xp(:,1),Xp(:,2), p.Results.mS, 'filled', 'bs', 'DisplayName', 'x|y=+1');

% Set axes properties
set(gca, 'FontSize', p.Results.fS);
set(gca, 'XLim', p.Results.u1l, 'YLim', p.Results.u2l);
gt = findobj(gcf, 'Type', 'Scatter');
legend(gt, {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex', 'Location', 'northwest')
xlabel('$$x_1$$', 'Interpreter', 'latex');
ylabel('$$x_2$$', 'Interpreter', 'latex');
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);

% Write figure to file
if ~isempty(p.Results.savnmS)
    saveas(gcf, p.Results.savnmS, 'epsc');
end

% Initialize figure for target data
figure() 
hold on

% Scatter target data
scatter(Zn(:,1),Zn(:,2), p.Results.mS, 'filled', 'kd', 'DisplayName', 'x|y=-1');
scatter(Zp(:,1),Zp(:,2), p.Results.mS, 'filled', 'kd', 'DisplayName', 'x|y=+1');

% Set axes properties
set(gca, 'FontSize', p.Results.fS);
set(gca, 'XLim', p.Results.u1l, 'YLim', p.Results.u2l);
gt = findobj(gcf, 'Type', 'Scatter');
legend(gt, {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex', 'Location', 'northwest')
xlabel('$$x_1$$', 'Interpreter', 'latex');
ylabel('$$x_2$$', 'Interpreter', 'latex');
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);

% Write figure to file
if ~isempty(p.Results.savnmT)
    saveas(gcf, p.Results.savnmT, 'epsc');
end

end
