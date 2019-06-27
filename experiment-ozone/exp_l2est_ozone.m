% Script to run sample selection bias experiments on UCI-ozone
close all;
clearvars;

addpath(genpath('../util'))

sav = true;
viz = false;
savnm = 'results/';

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool('local', 6)
end

%% Load pima dataset

load UCI-ozone

% Remove entries with nans
ixRem = sum(isnan(D),2)>0;
D(ixRem,:) = [];
y(ixRem) = [];

% Scale data
D = zscore(D,[],1);

% Reduce dim
dim = 10;
[~,DC,~] = pca(D);
DC = DC(:, 1:dim);
[M, F] = size(DC);

% Cast labels to {-1,+1}
labels = y;
labels(labels==0) = -1;

%%

% Number of folds
nF = 5;

% Number of repeats
nR = 1e3;

% Sample sizes
NS = 50;   % source data

% Importance weight estimator
iwT = 'gauss';
hyperparam = 0;

% Lambda range
Lambda = logspace(0,3,101);
nL = length(Lambda);

% Means for sampling probs
mu_S = -ones(1,dim);

% Target parameter changes
gamma = .5:.1:1;
nG = length(gamma);

% Weight truncation
max_weight = 10;

%% Experiment

% Preallocation
Vw = zeros(nG, nR);
Rh_S = zeros(nG, nL, nR);
Rh_W = zeros(nG, nL, nR);
Rh_C = zeros(nG, nL, nR);
Rh_M = zeros(nG, nL, nR);
Rh_T = zeros(nG, nL, nR);
lambda_S = zeros(nG, nR);
lambda_W = zeros(nG, nR);
lambda_C = zeros(nG, nR);
lambda_M = zeros(nG, nR);
lambda_T = zeros(nG, nR);
R_S = zeros(nG, nR);
R_W = zeros(nG, nR);
R_C = zeros(nG, nR);
R_M = zeros(nG, nR);
R_T = zeros(nG, nR);
wi = 0;

parfor r = 1:nR
    % Report progress
    if rem(r,nR/10)==1
        disp(['Repetition ' num2str(r) '/' num2str(nR)]);
    end
    
    for g = 1:nG
        
        % Find labels of each class
        ixn = find(labels==-1);
        ixp = find(labels==+1);
        
        % Source sampling probabilities
        Sigma_S = gamma(g)*cov(DC);
        pS = max(realmin, mvnpdf(DC, mu_S, Sigma_S));
        
        % Sample source data
        ixSn = datasample(ixn, NS/2, 'Replace', false, 'Weights', pS(ixn));
        ixSp = datasample(ixp, NS/2, 'Replace', false, 'Weights', pS(ixp));
        
        % Concatenate to datasets
        S = [DC(ixSn,:); DC(ixSp,:)];
        yS = [-ones(length(ixSn),1); ones(length(ixSp),1)];
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                % No weighting
                wi = ones(1,nV);
                
            case 'gauss'
                
                % Ratio of Gaussian distributions
                wi = iw_Gauss(S, DC, 'l2', 0);
                
            case 'kde'
                % Sorted distance from source to target
                nnd = sort(pdist2(S, DC), 2, 'ascend');

                % Average distance to 5-th nearest neighbour
                hyperparam = mean(nnd(:, 5), 1);
                
                % Kernel density estimator
                wi = iw_kde(S, DC, ...
                            'bw', hyperparam, ...
                            'self_normalize', false);
            case 'kmm'  
                
                % Sorted distance from source to target
                nnd = sort(pdist2(S, DC), 2, 'ascend');

                % Average distance to 5-th nearest neighbour
                hyperparam = mean(nnd(:, 5), 1);
            
                % Kernel mean matching
                wi = iw_KMM(S, DC, ...
                            'theta', hyperparam, ...
                            'B', Inf, ...
                            'eps', 1e-6);
            case 'kliep'
                
                % Sorted distance from source to target
                nnd = sort(pdist2(S, DC), 2, 'ascend');

                % Average distance to 5-th nearest neighbour
                hyperparam = mean(nnd(:, 5), 1);
            
                % Kullback-Leibler Importance-Estimation Procedure
                wi = iw_KLIEP(S, DC, 'sigma', hyperparam);
                
            case 'nn'
                % Nearest-neighbour-based weighting
                wi = iw_NNeW(S, DC, 0, realmax, 'Laplace', 1);
                
            otherwise
                error('Unknown importance weight estimator');
        end
        
        % Augment data
        Sa = [S ones(NS,1)];
        Ta = [DC ones(M, 1)];
        
        % Sample folds
        foldsp = randsample(1:nF, length(ixSp), true);
        foldsn = randsample(1:nF, length(ixSn), true);
        
        % Populate fold index vector
        folds = zeros(NS,1);
        folds(1:length(ixSp)) = foldsp;
        folds(length(ixSp)+1:end) = foldsn;
        
        % Preallocate loss vectors
        L_S = zeros(NS, nL);
        L_W = zeros(NS, nL);
        L_C = zeros(NS, nL);
        L_M = zeros(NS, nL);
        L_T = zeros(M, nL);
        
        for f = 1:nF
            
            % Split source data into training and validation
            Xa = Sa(f ~= folds, :); 
            yX = yS(f ~= folds);
            
            Va = Sa(f == folds, :);
            yV = yS(f == folds);
            
            % Split weights for validation
            wi_f = wi(f == folds);
            wi_g = wi(f ~= folds);
            
            for l = 1:nL
                
                % Train classifier
                theta = (Xa'*diag(wi_g)*Xa + Lambda(l)*eye(dim+1))\(Xa'*diag(wi_g)*yX);
                
                % Compute empirical risks
                L_S(f==folds, l) = (Va*theta - yV).^2;
                L_W(f==folds, l) = (Va*theta - yV).^2 .* wi_f;
                L_M(f==folds, l) = (Va*theta - yV).^2 .* min(max_weight, wi_f);
                L_T(:, l) = (Ta*theta - labels).^2;
                
                % Compute beta
                weighted_loss = (Va*theta - yV).^2 .*wi_f;
                a_i = weighted_loss - mean(weighted_loss, 1);
                b_i = wi_f - mean(wi_f);
                beta = sum( a_i .* b_i, 1) ./ sum( b_i.^2, 1);
                
                % Compute controlled risk
                L_C(f==folds, l) = (Va*theta - yV).^2 .* wi_f - beta.*(wi_f - 1);
                
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
        
        % Convert to values instead of indices
        lambda_S(g, r) = Lambda(lambda_S(g,r));
        lambda_W(g, r) = Lambda(lambda_W(g,r));
        lambda_C(g, r) = Lambda(lambda_C(g,r));
        lambda_M(g, r) = Lambda(lambda_M(g,r));
        lambda_T(g, r) = Lambda(lambda_T(g,r));
        
        % Classifier on whole set
        eta = @(l2) (Sa'*diag(wi)*Sa + l2*eye(dim+1))\(Sa'*diag(wi)*yS);
        
        % Compute target risk
        RT = @(theta) mean( (Ta*theta - labels).^2, 1);
        
        % True target risks for selected lambda's
        R_S(g,r) = RT( eta(lambda_S(g, r)));
        R_W(g,r) = RT( eta(lambda_W(g, r)));
        R_C(g,r) = RT( eta(lambda_C(g, r)));
        R_M(g,r) = RT( eta(lambda_M(g, r)));
        R_T(g,r) = RT( eta(lambda_T(g, r)));
        
        % Store weights
        Vw(g, r) = nanvar(wi);
        
    end
end 

% Write results to file
di = 1; 
while exist([savnm 'exp_l2est_ozone_iw-' iwT '_hyperparam' num2str(hyperparam) '_' num2str(di) '.mat'], 'file')
    di = di+1; 
end
fn = [savnm 'exp_l2est_ozone_iw-' iwT '_hyperparam' num2str(hyperparam) '_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'nR', 'gamma', 'Lambda', 'Vw', 'hyperparam', ...
    'R_S', 'R_W', 'R_C', 'R_T', 'R_M', ...
    'lambda_S', 'lambda_W', 'lambda_C', 'lambda_T', 'lambda_M', ...
    'Rh_T', 'Rh_S', 'Rh_W', 'Rh_C', 'Rh_T', 'Rh_M');

% Close parpool
delete(gcp('nocreate'))
