function viz_dists_2DG(pXy,pyX,varargin)
% Contour plot the class-conditional and class-posterior distributions

% Parse arguments
p = inputParser;
addOptional(p, 'lW', 2);
addOptional(p, 'fS', 15);
addOptional(p, 'nU', 101);
addOptional(p, 'u1l', [-4 +4]);
addOptional(p, 'u2l', [-4 +4]);
addOptional(p, 'mu_S', [0 0]);
addOptional(p, 'mu_T', [0 0]);
addOptional(p, 'Sigma_S', eye(2));
addOptional(p, 'Sigma_T', eye(2));
addOptional(p, 'savnm', {'','','',''});
parse(p, varargin{:});

% Span grid
u1 = linspace(p.Results.u1l(1),p.Results.u1l(2),p.Results.nU);
u2 = linspace(p.Results.u2l(1),p.Results.u2l(2),p.Results.nU);
[tu1,tu2] = meshgrid(u1,u2);

% Initialize figure
figure();
hold on

% Plot conditional distributions of source domain
contour(u1',u2',reshape(pXy(+1, tu1(:), tu2(:), p.Results.mu_S, p.Results.Sigma_S), [p.Results.nU p.Results.nU]), 'Color', 'b', 'LineWidth', p.Results.lW);
contour(u1',u2',reshape(pXy(-1, tu1(:), tu2(:), p.Results.mu_S, p.Results.Sigma_S), [p.Results.nU p.Results.nU]), 'Color', 'r', 'LineWidth', p.Results.lW);

% Set axes properties
set(gca, 'FontSize', p.Results.fS)
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
legend({'p_S(x | y = +1)', 'p_S(x | y =  -1)'}, 'FontSize', p.Results.fS-5)
title(['Source conditional distributions' newline '$$p_{\cal S}(x|y)$$'], 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10)

% Write figure to file
saveas(gcf, p.Results.savnm{1}, 'epsc');

% Initialize figure
figure();
hold on

% Plot posterior distributions
contour(u1',u2',reshape(pyX(+1,tu1(:),tu2(:)), [p.Results.nU p.Results.nU]), 'Color', 'b', 'LineWidth', p.Results.lW);
contour(u1',u2',reshape(pyX(-1,tu1(:),tu2(:)), [p.Results.nU p.Results.nU]), 'Color', 'r', 'LineWidth', p.Results.lW);
contour(u1',u2',reshape(pyX(+1,tu1(:),tu2(:)), [p.Results.nU p.Results.nU]) - reshape(pyX(-1,tu1(:),tu2(:)), [p.Results.nU p.Results.nU]), [0,0], 'Color', 'k', 'LineWidth', 3)

% Set axes properties
set(gca, 'FontSize', p.Results.fS)
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
title(['Posterior distributions' newline '$$p_{\cal S}(y|x) = p_{\cal T}(y|x)$$'], 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10)
legend({'y = +1 | x', 'y =  -1 | x'}, 'FontSize', p.Results.fS-5)

% Write figure to file
saveas(gcf, p.Results.savnm{2}, 'epsc');

% Initialize figure
figure();
hold on

% Plot conditional distributions of target domain
contour(u1',u2',reshape(pXy(+1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T), [p.Results.nU p.Results.nU]), 'Color', 'b', 'LineWidth', p.Results.lW);
contour(u1',u2',reshape(pXy(-1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T), [p.Results.nU p.Results.nU]), 'Color', 'r', 'LineWidth', p.Results.lW);

% Set axes properties
set(gca, 'FontSize', p.Results.fS)
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
title(['Conditional distributions' newline '$$p_{\cal T}(x|y)$$'], 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10)
legend({'p_T(x | y = +1)', 'p_T(x | y =  -1)'}, 'FontSize', p.Results.fS-5)

% Write figure to file
saveas(gcf, p.Results.savnm{3}, 'epsc');

% Initialize figure
figure()
hold on

% Marginalize out y
pX = pXy(+1,tu1(:),tu2(:),p.Results.mu_S,p.Results.Sigma_S)./2 + pXy(-1,tu1(:),tu2(:),p.Results.mu_S,p.Results.Sigma_S)./2;
pZ = pXy(+1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T)./2 + pXy(-1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T)./2;

% Plot marginal distributions of each domain
contour(u1',u2',reshape(pX, [p.Results.nU p.Results.nU]), 'k');
contour(u1',u2',reshape(pZ, [p.Results.nU p.Results.nU]), 'k-.');

% Set axes properties
set(gca, 'FontSize', p.Results.fS)
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10);
title(['Data marginal distributions' newline '$$p(x)$$'], 'Interpreter', 'latex', 'FontSize', p.Results.fS + 10)
legend({'p_S(x)','p_T(x)'});

% Write figure to file
saveas(gcf, p.Results.savnm{4}, 'epsc');

end
