function viz_conditionals(pXy,Zn,Zp,Sn,Sp,varargin)
% Contour plot the class-conditional distributions

% Parse arguments
p = inputParser;
addOptional(p, 'mS', 120);
addOptional(p, 'fS', 20);
addOptional(p, 'fh', '');
addOptional(p, 'nU', 101);
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

u1 = linspace(p.Results.u1l(1),p.Results.u1l(2),p.Results.nU);
u2 = linspace(p.Results.u2l(1),p.Results.u2l(2),p.Results.nU);
[tu1,tu2] = meshgrid(u1,u2);

if ~isempty(p.Results.fh); p.Results.fh; end
if p.Results.marg; nSP = 3; else; nSP = 2; end

if ~isempty(p.Results.fh)
    subplot(1,nSP,1);
else
    figure()
end  
hold on

scatter(Sn(:,1),Sn(:,2), p.Results.mS, 'filled', 'rs', 'DisplayName', 'x|y=-1');
scatter(Sp(:,1),Sp(:,2), p.Results.mS, 'filled', 'bs', 'DisplayName', 'x|y=+1');
contour(u1',u2',reshape(pXy(+1,tu1(:),tu2(:),p.Results.mu_S,p.Results.Sigma_S), [p.Results.nU p.Results.nU]), 'LineColor', 'b', 'DisplayName', 'x|y=+1')
contour(u1',u2',reshape(pXy(-1,tu1(:),tu2(:),p.Results.mu_S,p.Results.Sigma_S), [p.Results.nU p.Results.nU]), 'LineColor', 'r', 'DisplayName', 'x|y=-1')
set(gca, 'FontSize', p.Results.fS);
gt = findobj(gcf, 'Type', 'Scatter');
legend(gt, {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex', 'Location', 'northwest')
xlabel('$$x_1$$', 'Interpreter', 'latex');
ylabel('$$x_2$$', 'Interpreter', 'latex');
set(gcf, 'Color', 'w', 'Position', [100 100 1600 600]);

if ~isempty(p.Results.savnmS)
    saveas(gcf,p.Results.savnmS, 'epsc');
end

%%

if ~isempty(p.Results.fh)
    subplot(1,nSP,2);
else
    figure()
end  
hold on

scatter(Zn(:,1),Zn(:,2), p.Results.mS, 'filled', 'kd', 'DisplayName', 'x|y=-1');
scatter(Zp(:,1),Zp(:,2), p.Results.mS, 'filled', 'kd', 'DisplayName', 'x|y=+1');
contour(u1',u2',reshape(pXy(+1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T), [p.Results.nU p.Results.nU]), 'LineColor', 'k', 'DisplayName', 'z|y=+1')
contour(u1',u2',reshape(pXy(-1,tu1(:),tu2(:),p.Results.mu_T,p.Results.Sigma_T), [p.Results.nU p.Results.nU]), 'LineColor', 'k', 'DisplayName', 'z|y=+1')
set(gca, 'FontSize', p.Results.fS);
gt = findobj(gcf, 'Type', 'Scatter');
legend(gt, {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex', 'Location', 'northwest')
xlabel('$$x_1$$', 'Interpreter', 'latex');
ylabel('$$x_2$$', 'Interpreter', 'latex');
set(gcf, 'Color', 'w', 'Position', [100 100 1600 600]);

if ~isempty(p.Results.savnmT)
    saveas(gcf,p.Results.savnmT, 'epsc');
end

end
