% Script to run some covariate shift experiments
close all;
clearvars;

addpath(genpath('../util'))

save_figs = true;
san = false;
viz = false;
savnm = 'results/';

if ~exist('results', 'dir'); mkdir('results'); end

%%

% Number of folds
nF = 5;

% Number of repeats
nR = 1e2;

% Sample sizes
N = 50;   % source data
M = 1000;  % target training

% Importance weight estimator
iwT = 'true';

% Lambda range
Lambda = logspace(-3,6,201);
nL = length(Lambda);

% Source parameters
mu_S = [-1 0];
Sigma_S = [1 0; 0 1];

% Target parameters
mu_T = [0 0];
Sigma_T = eye(2);

% Target parameter changes
rho = 1./2;
shifttype = 'var';
delta = 2.^[-1.5:.25:0];
nD = length(delta);


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
W = zeros(nD,N,nR);

Rh_S = zeros(nD, nL, nR);
Rh_W = zeros(nD, nL, nR);
Rh_C = zeros(nD, nL, nR);
Rh_T = zeros(nD, nL, nR);

lambda_S = NaN(nD, nR);
lambda_W = NaN(nD, nR);
lambda_C = NaN(nD, nR);
lambda_T = NaN(nD, nR);

R_S = zeros(nD, nR);
R_W = zeros(nD, nR);
R_C = zeros(nD, nR);
R_T = zeros(nD, nR);

for d = 1:nD
    disp(['Change in parameter: ' num2str(delta(d))]);
    
    % New parameters
    Sigma_S = delta(d)*eye(2);
    
    % Define grid for rejection sampling
    nU = 101;
    u1lT = [-6 4];
    u2lT = [-4 4];
    u1lS = [-6 4];
    u2lS = [-4 4];
    
    % Helper functions
    pX_yn = @(x1,x2) pXy(-1,x1,x2,mu_S,Sigma_S);
    pX_yp = @(x1,x2) pXy(+1,x1,x2,mu_S,Sigma_S);
    pZ_yn = @(x1,x2) pZy(-1,x1,x2,mu_T,Sigma_T);
    pZ_yp = @(x1,x2) pZy(+1,x1,x2,mu_T,Sigma_T);
    
    for r = 1:nR
        
        % Report progress over repetitions
        if (rem(r,nR./10)==1)
            fprintf('At repetition \t%i/%i\n', r, nR)
        end
        
        % Rejection sampling of target final risk validation data
        MT = 1.2/sqrt(det(2*pi*Sigma_T));
        
        % Rejection sampling of target data
        Zn = sampleDist2(pZ_yn,MT,round(M.*py(-1)),u1lT, u2lT);
        Zp = sampleDist2(pZ_yp,MT,round(M.*py(+1)),u1lT, u2lT);
        
        % Rejection sampling of source data
        MS = 1.2/sqrt(det(2*pi*Sigma_S));
        Sn = sampleDist2(pX_yn,MS,round(N.*py(-1)),u1lS, u2lS);
        Sp = sampleDist2(pX_yp,MS,round(N.*py(+1)),u1lS, u2lS);
        
        % Concatenate to datasets
        Z = [Zn; Zp];
        S = [Sn; Sp];
        yZ = [-ones(size(Zn,1),1); ones(size(Zp,1),1)];
        yS = [-ones(size(Sn,1),1); ones(size(Sp,1),1)];
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                W(d,:,r) = ones(1,nV);
            case 'true'
                W(d,:,r) = pZ(S(:,1),S(:,2),mu_T,Sigma_T) ./ pX(S(:,1),S(:,2),mu_S,Sigma_S);
            case 'gauss'
                W(d,:,r) = iw_Gauss(S,Z,0,realmax);
            otherwise
                error('Unknown importance weight estimator');
        end
        
        % Augment data
        Z = [Z ones(M, 1)];
        S = [S ones(N, 1)];
        
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
        L_T = zeros(M, nL);
        
        for f = 1:nF
            
            % Split source data into training and validation
            X = S(f ~= folds, :);
            yX = yS(f ~= folds);
            
            V = S(f == folds, :);
            yV = yS(f == folds);
            
            % Split weights for validation
            WV = W(d, f == folds, r)';
            
            for l = 1:nL
                
                % Train classifier
                theta = (X'*X + Lambda(l)*eye(3))\(X'*yX);
                
                % Compute empirical risks
                L_S(f==folds, l) = (V*theta - yV).^2;
                L_W(f==folds, l) = (V*theta - yV).^2 .* WV;
                L_T(:, l) = (Z*theta - yZ).^2;
                
                % Compute beta
                weighted_loss = (V*theta - yV).^2 .*WV;
                a_i = weighted_loss - mean(weighted_loss, 1);
                b_i = WV - mean(WV);
                beta = sum( a_i .* b_i, 1) ./ sum( b_i.^2, 1);
                
                % Compute controlled risk
                L_C(f==folds, l) = (V*theta - yV).^2 .* WV - beta.*(WV - 1);
                
            end
        end
        
        % Compute empirical risks
        Rh_S(d, :, r) = mean(L_S, 1)';
        Rh_W(d, :, r) = mean(L_W, 1)';
        Rh_C(d, :, r) = mean(L_C, 1)';
        Rh_T(d, :, r) = mean(L_T, 1)';
        
        % Find best lambda minima
        [~, lambda_S(d, r)] = min(Rh_S(d, :, r), [], 2);
        [~, lambda_W(d, r)] = min(Rh_W(d, :, r), [], 2);
        [~, lambda_C(d, r)] = min(Rh_C(d, :, r), [], 2);
        [~, lambda_T(d, r)] = min(Rh_T(d, :, r), [], 2);
        
        % Classifier on whole set
        eta = @(l) (S'*S + Lambda(l)*eye(3))\(S'*yS);
        
        % Compute target risk
        RT = @(theta) mean( (Z*theta - yZ).^2, 1);
        
        % True target risks for selected lambda's
        R_S(d, r) = RT( eta(lambda_S(d,r)) );
        R_W(d, r) = RT( eta(lambda_W(d,r)) );
        R_C(d, r) = RT( eta(lambda_C(d,r)) );
        R_T(d, r) = RT( eta(lambda_T(d,r)) );

    end
end


di = 1; while exist([savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm 'exp_l2est_2DG_iw-' iwT '_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'delta', 'Lambda', 'W', 'R_S', 'R_W', 'R_C', 'R_T', ...
    'lambda_S', 'lambda_W', 'lambda_C', 'lambda_T', ...
    'Rh_T', 'Rh_S', 'Rh_W', 'Rh_C', 'Rh_T');


