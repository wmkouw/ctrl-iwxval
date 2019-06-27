% Script to run some covariate shift experiments
close all;
clearvars;

addpath(genpath('../util'))

save_figs = true;
san = false;
viz = false;
savnm = 'results/';

if ~exist('results', 'dir')
    mkdir('results'); 
end

%% Multi-thread options

if isempty(gcp('nocreate'))
    parpool('local', 3)
end

%%

% Number of folds
nF = 5;

% Number of repeats
nR = 1e5;

% Sample sizes
N = 50;   % source data
M = 1000;  % target training

% Importance weight estimator
iwT = 'gauss';

% Lambda range
Lambda = logspace(0,3,201);
nL = length(Lambda);

% Source parameters
mu_S = [-1 0];
Sigma_S = [1 0; 0 1];

% Target parameters
mu_T = [0 0];
Sigma_T = eye(2);

% Target parameter changes
gamma = .5:.1:1;
nG = length(gamma);

% Truncation constant
max_weight = 10;

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

% Preallocate variables
Vw = zeros(nG, nR);
Mw = zeros(nG, nR);

Rh_S = zeros(nG, nL, nR);
Rh_W = zeros(nG, nL, nR);
Rh_C = zeros(nG, nL, nR);
Rh_M = zeros(nG, nL, nR);
Rh_T = zeros(nG, nL, nR);

lambda_S = NaN(nG, nR);
lambda_W = NaN(nG, nR);
lambda_C = NaN(nG, nR);
lambda_M = NaN(nG, nR);
lambda_T = NaN(nG, nR);

R_S = zeros(nG, nR);
R_W = zeros(nG, nR);
R_C = zeros(nG, nR);
R_M = zeros(nG, nR);
R_T = zeros(nG, nR);

for g = 1:nG
    disp(['Change in parameter: ' num2str(gamma(g))]);
    
    % New parameters
    Sigma_S = gamma(g)*eye(2);
    
    % Define grid for rejection sampling
    nU = 101;
    u1lT = [-6 4];
    u2lT = [-4 4];
    u1lS = [-6 4];
    u2lS = [-4 4];
    
    % Helper functions
    pS_yn = @(x1,x2) pXy(-1,x1,x2,mu_S,Sigma_S);
    pS_yp = @(x1,x2) pXy(+1,x1,x2,mu_S,Sigma_S);
    pT_yn = @(x1,x2) pZy(-1,x1,x2,mu_T,Sigma_T);
    pT_yp = @(x1,x2) pZy(+1,x1,x2,mu_T,Sigma_T);
    
    parfor r = 1:nR
        
        % Report progress over repetitions
        if (rem(r,nR./10)==1)
            fprintf('At repetition \t%i/%i\n', r, nR)
        end
        
        % Rejection sampling of target final risk validation data
        MT = 1.2/sqrt(det(2*pi*Sigma_T));
        
        % Rejection sampling of target data
        Tn = sampleDist2D(pT_yn,MT,round(M.*py(-1)),u1lT, u2lT);
        Tp = sampleDist2D(pT_yp,MT,round(M.*py(+1)),u1lT, u2lT);
        
        % Rejection sampling of source data
        MS = 1.2/sqrt(det(2*pi*Sigma_S));
        Sn = sampleDist2D(pS_yn,MS,round(N.*py(-1)),u1lS, u2lS);
        Sp = sampleDist2D(pS_yp,MS,round(N.*py(+1)),u1lS, u2lS);
        
        % Concatenate to datasets
        T = [Tn; Tp];
        S = [Sn; Sp];
        yT = [-ones(size(Tn,1),1); ones(size(Tp,1),1)];
        yS = [-ones(size(Sn,1),1); ones(size(Sp,1),1)];
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                W = ones(1,nV);
            case 'true'
                W = pZ(S(:,1),S(:,2),mu_T,Sigma_T) ./ pX(S(:,1),S(:,2),mu_S,Sigma_S);
            case 'gauss'
                W = iw_Gauss(S, T, 0, realmax);
            case 'kliep'
                % Bandwidth selection
                [~, ~, bw] = ksdensity(S, T);
                
                W = iw_KLIEP(S, T, 'sigma', mean(bw), 'verbose', false);
            case 'kmm'
                
                W = iw_KMM(S, T, 'B', Inf, 'theta', -1, 'eps', 1e-6);
                
            case 'nn'
                W = iw_NNeW(S, T, 'Laplace', true);
            otherwise
                error('Unknown importance weight estimator');
        end
        
        % Compute variance of weights
        Vw(g, r) = nanvar(W);
        Mw(g, r) = max(W(:));
        
        % Augment data
        Ta = [T ones(M, 1)];
        Sa = [S ones(N, 1)];
        
        % Class indices
        ixp = find(yS==+1);
        ixn = find(yS==-1);
        
        % Sample folds
        foldsp = randsample(1:nF, length(ixp), true);
        foldsn = randsample(1:nF, length(ixn), true);
        
        % Populate fold index vector
        folds = zeros(N,1);
        folds(ixp) = foldsp;
        folds(ixn) = foldsn;
        
        % Preallocate loss vectors
        L_S = zeros(N, nL);
        L_W = zeros(N, nL);
        L_C = zeros(N, nL);
        L_M = zeros(N, nL);
        L_T = zeros(M, nL);
        
        for f = 1:nF
            
            % Split source data into training and validation
            X = Sa(f ~= folds, :);
            yX = yS(f ~= folds);
            
            V = Sa(f == folds, :);
            yV = yS(f == folds);
            
            % Split weights
            Wf = W(f == folds);
            Wg = W(f ~= folds);
            
            for l = 1:nL
                
                % Train classifier
                theta = (X'*diag(Wg)*X + Lambda(l)*eye(3))\(X'*diag(Wg)*yX);
                
                % Compute empirical risks
                L_S(f==folds, l) = (V*theta - yV).^2;
                L_W(f==folds, l) = (V*theta - yV).^2 .* Wf;
                L_M(f==folds, l) = (V*theta - yV).^2 .* min(max_weight, Wf);
                L_T(:, l) = (Ta*theta - yT).^2;
                
                % Compute beta
                weighted_loss = (V*theta - yV).^2 .* Wf';
                a_i = weighted_loss - mean(weighted_loss, 1);
                b_i = Wf - mean(Wf);
                beta = sum( a_i .* b_i, 1) ./ sum( b_i.^2, 1);
                
                % Compute controlled risk
                L_C(f==folds, l) = (V*theta - yV).^2 .* Wf - beta'.*(Wf - 1);
                
            end
        end
        
        % Compute empirical risks
        Rh_S(g, :, r) = mean(L_S, 1)';
        Rh_W(g, :, r) = mean(L_W, 1)';
        Rh_C(g, :, r) = mean(L_C, 1)';
        Rh_M(g, :, r) = mean(L_M, 1)';
        Rh_T(g, :, r) = mean(L_T, 1)';
        
        % Find best lambda minima
        [~, lambda_S(g, r)] = min(Rh_S(g, :, r), [], 2);
        [~, lambda_W(g, r)] = min(Rh_W(g, :, r), [], 2);
        [~, lambda_C(g, r)] = min(Rh_C(g, :, r), [], 2);
        [~, lambda_M(g, r)] = min(Rh_M(g, :, r), [], 2);
        [~, lambda_T(g, r)] = min(Rh_T(g, :, r), [], 2);
        
        % Classifier on whole set
        eta = @(l) (Sa'*diag(W)*Sa + Lambda(l)*eye(3))\(Sa'*diag(W)*yS);
        
        % Compute target risk
        RT = @(theta) mean( (Ta*theta - yT).^2, 1);
        
        % True target risks for selected lambda's
        R_S(g, r) = RT( eta(lambda_S(g,r)) );
        R_W(g, r) = RT( eta(lambda_W(g,r)) );
        R_C(g, r) = RT( eta(lambda_C(g,r)) );
        R_M(g, r) = RT( eta(lambda_M(g,r)) );
        R_T(g, r) = RT( eta(lambda_T(g,r)) );

    end
end

% Write results to file
di = 1; while exist([savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'nR', 'gamma', 'Lambda', 'Vw', 'Mw', ...
    'lambda_S', 'lambda_W', 'lambda_C', 'lambda_T', 'lambda_M', ...
    'R_S', 'R_W', 'R_C', 'R_T', 'R_M', ...
    'Rh_T', 'Rh_S', 'Rh_W', 'Rh_C', 'Rh_T', 'Rh_M', '-v7.3');

% Close parpool
delete(gcp('nocreate'))
