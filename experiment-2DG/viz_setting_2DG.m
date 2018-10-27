% Script to run some covariate shift experiments
close all;
clearvars;

addpath(genpath('../util'))

save_figs = false;
san = false;
viz = false;

savnm = {'viz/pSxy.eps','viz/pyx.eps', 'viz/pTxy.eps', 'viz/px.eps'};

%%

% Sample sizes
N = 50;
M = 1000;

% Source parameters
mu_S = [-1 0];
Sigma_S = [1 0; 0 1];

% Target parameters
mu_T = [0 0];
Sigma_T = eye(2);

% Target parameter changes
rho = 1./2;
delta = 2.^-2;
nD = length(delta);

mS = 100;
fS = 20;
lW = 5;

%% Equal priors and equal class-posteriors

Y = [-1,+1];
nK = length(Y);

% Priors
py = @(y) 1./2;

% Class-posteriors
pyX = @(y,x1,x2) (y<0)+y*mvncdf(-[x1 x2]*2, [0 0]);

%%% Domain specific

% Source marginal
pX = @(x1,x2,mu_S,Si_S) mvnpdf([x1 x2], mu_S, Si_S);

% Source class-conditional likelihoods
pXy = @(y,x1,x2,mu_S,Si_S) (pyX(y,x1,x2) .* pX(x1,x2,mu_S,Si_S))./py(y);

% Target marginal
pZ = @(x1,x2,mu_T,Si_T) mvnpdf([x1 x2], mu_T, Si_T);

% Target class-conditional likelihoods
pZy = @(y,x1,x2,mu_T,Si_T) (pyX(y,x1,x2) .* pZ(x1,x2,mu_T,Si_T))./py(y);

%%

for d = 1:nD
    
    % Report current parameter
    disp(['Change in parameter: ' num2str(delta(d))]);
    
    % Set to sigma
    Sigma_S = delta(d)*eye(2);
    
    % Define grid for rejection sampling
    nU = 101;
    u1lT = [-4 3];
    u2lT = [-3 3];
    u1lS = [-4 3];
    u2lS = [-3 3];
    
    % Helper functions
    pX_yn = @(x1,x2) pXy(-1,x1,x2,mu_S,Sigma_S);
    pX_yp = @(x1,x2) pXy(+1,x1,x2,mu_S,Sigma_S);
    pZ_yn = @(x1,x2) pZy(-1,x1,x2,mu_T,Sigma_T);
    pZ_yp = @(x1,x2) pZy(+1,x1,x2,mu_T,Sigma_T);
    
    % Rejection sampling of target final risk validation data
    MT = 1.2/sqrt(det(2*pi*Sigma_T));
    
    % Rejection sampling of target data
    Zn = sampleDist2(pZ_yn,MT,round(M.*py(-1)),u1lT, u2lT);
    Zp = sampleDist2(pZ_yp,MT,round(M.*py(+1)),u1lT, u2lT);
    
    % Rejection sampling of source data
    MS = 1.2/sqrt(det(2*pi*Sigma_S));
    Xn = sampleDist2(pX_yn,MS,round(N.*py(-1)),u1lS, u2lS);
    Xp = sampleDist2(pX_yp,MS,round(N.*py(+1)),u1lS, u2lS);
    
    % Concatenate to datasets
    Z = [Zn; Zp];
    X = [Xn; Xp];
    yZ = [-ones(size(Zn,1),1); ones(size(Zp,1),1)];
    yX = [-ones(size(Xn,1),1); ones(size(Xp,1),1)];
    
    % Visualize distributions
    viz_dists_2DG(pZy,pyX, 'mu_S', mu_S, 'Sigma_S', Sigma_S, ...
        'mu_T', mu_T, 'Sigma_T', Sigma_T, 'u1l', u1lS, 'u2l', u2lS, ...
        'savnm', savnm, 'fS', fS);

    % Visualize conditionals
    viz_scatter_2DG(Zn,Zp,Xn,Xp,'mu_S', mu_S, 'Sigma_S', Sigma_S, ...
        'mu_T', mu_T, 'Sigma_T', Sigma_T, 'u1l', u1lS, 'u2l', u2lS, ...
        'savnmS', 'viz/2DG_scatter_S.eps', ...
        'savnmT', 'viz/2DG_scatter_T.eps', 'fS', fS);
    
end





