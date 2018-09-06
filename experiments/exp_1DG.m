% Script to run some covariate shift experiments
close all;
clearvars;

addpath(genpath('../util'))

san = false;
viz = false;
savnm = 'results/';

%%

% Number of folds
nF = 2;

% Sample sizes
NS = 40;   % source data
NZ = 20;    % target training

% Number of times split
nS = 1;

% Number of repeats
nR = 1e4;

% Importance weight estimator
shifttype = 'var';
iwT = 'true';

% Basis function
ktype = 'lin';
switch ktype
    case 'lin'
        K = @(x,z,d) x'*z;
    case 'pol'
        K = @(x,z,d) (x*z'+0).^d;
    case 'rbf'
        K = @(x,z,l) exp(-pdist2(x,z,'squaredeuclidean')./l);
end

% Lambda range
nL = 100;
if strcmp(ktype, 'rbf')
    Lambda = linspace(-0.2,1,nL);
else
%     Lambda = linspace(0.01,10,nL)*NS;
     Lambda = [-0.2:.01:1]*NS;
     nL = length(Lambda);
end

% Kernel bandwidth
nu = [1];
nN = 1;

% Source parameters
mu_S = 0;
sigma2_S = 1;

% Target parameters
mu_T = 0;
sigma2_T = 1;

% Target parameter changes
delta = 2.^[-3:2];
nD = length(delta);

%% 2D grid

nU = 201;
ul = [-6 +6];
u1 = linspace(ul(1),ul(2),nU);

%% Equal priors and equal class-posteriors

% Priors
pY = @(y) 1./2;

% Class-posteriors
pYS = @(y,x) normcdf(y*x);

% Source marginal
pS = @(x,mu_S,si2_S) normpdf(x, mu_S, sqrt(si2_S));

% Source class-conditional likelihoods
pSY = @(y,x,mu_S,si_S) (pYS(y,x) .* pS(x,mu_S,si_S))./pY(y);

% Target marginal
pT = @(x,mu_T,si2_T) normpdf(x, mu_T, sqrt(si2_T));

% Target class-conditional likelihoods
pTY = @(y,x,mu_T,si_T) (pYS(y,x) .* pT(x,mu_T,si_T))./pY(y);

%%

iw = NaN(nD,NS,nR);
% beta = NaN(nL,nN,nD,nR);
MSE.V = NaN(nL,nN,nD,nR);
MSE.W = NaN(nL,nN,nD,nR);
MSE.B = NaN(nL,nN,nD,nR);
MSE.C = NaN(nL,nN,nD,nR);
MSE.Z = NaN(nL,nN,nD,nR);
sVh.V = NaN(nL,nN,nD,nR);
sVh.W = NaN(nL,nN,nD,nR);
sVh.C = NaN(nL,nN,nD,nR);
sVh.B = NaN(nL,nN,nD,nR);
sVh.Z = NaN(nL,nN,nD,nR);
minl.V = NaN(nD,nR);
minl.W = NaN(nD,nR);
minl.C = NaN(nD,nR);
minl.B = NaN(nD,nR);
minl.Z = NaN(nD,nR);
minb.V = NaN(nD,nR);
minb.W = NaN(nD,nR);
minb.C = NaN(nD,nR);
minb.B = NaN(nD,nR);
minb.Z = NaN(nD,nR);
RTh.V = NaN(nD,nR);
RTh.W = NaN(nD,nR);
RTh.C = NaN(nD,nR);
RTh.B = NaN(nD,nR);
RTh.Z = NaN(nD,nR);

dVQ = NaN(nD,nR);
dVT = NaN(nD,nR);
dVZ = NaN(nD,nR);
dVS = NaN(nD,nR);

for d = 1:nD
    disp(['Change in parameter: ' num2str(delta(d))]);
    
    % Current set of gamma
    sigma2_T = delta(d);
    
    % Target rejection sampling limits
    ulT = delta(d)*ul;
    
    % Helper functions
    pS_yn = @(x) pSY(-1,x,mu_S,sigma2_S);
    pS_yp = @(x) pSY(+1,x,mu_S,sigma2_S);
    pT_yn = @(x) pTY(-1,x,mu_T,sigma2_T);
    pT_yp = @(x) pTY(+1,x,mu_T,sigma2_T);
    
    for r = 1:nR
        
        % Rejection sampling of target validation data
        MT = 1./sqrt(2*pi*sigma2_T);
        Zy_n = sampleDist(pT_yn,MT,round(NZ.*pY(-1)),ulT);
        Zy_p = sampleDist(pT_yp,MT,round(NZ.*pY(+1)),ulT);
        
        % Rejection sampling of source data
        MS = 1./sqrt(2*pi*sigma2_S);
        Sy_n = sampleDist(pS_yn,MS,round(NS.*pY(-1)),ul);
        Sy_p = sampleDist(pS_yp,MS,round(NS.*pY(+1)),ul);
        
        % Concatenate to datasets
        Z = [Zy_n; Zy_p];
        S = [Sy_n; Sy_p];
        yZ = [-ones(size(Zy_n,1),1); ones(size(Zy_p,1),1)];
        yS = [-ones(size(Sy_n,1),1); ones(size(Sy_p,1),1)];
        
        % Sanity check
        dVZ(d,r) = var(Z) - sigma2_T;
        dVS(d,r) = var(S) - sigma2_S;
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                iw(d,:,r) = ones(1,nV);
            case 'true'
                iw(d,:,r) = pT(S,mu_T,sigma2_T) ./ pS(S,mu_S,sigma2_S);
            case 'gauss'
                iw(d,:,r) = iw_Gauss(V,T);
            case 'kde'
                iw(d,:,r) = iw_kde(V,T);
            case 'kmm'
                iw(d,:,r) = iw_KMM(V,T, 'theta', 1);
            case 'kliep'
                iw(d,:,r) = iw_KLIEP(V,T,0,realmax);
            case 'nnew'
                iw(d,:,r) = iw_NNeW(V,T,0,realmax, 'Laplace', 1);
            otherwise
                error('Unknown importance weight estimator');
        end
        
        % Augment with bias, if linear
        if strcmp(ktype, 'lin')
            S = [S ones(size(S))];
            Z = [Z ones(size(Z))];
        end
        
        % Loop over lambda
        for l = 1:nL
            % Loop over bandwidth
            for b = 1:nN
                
                lV = NaN(nS,NS);
                lZ = NaN(nS,NZ*nF);
                iwf = NaN(nS,NS);
                
                % Loop over number of times split
                for s = 1:nS
                    
                    % Class indices
                    ixp = find(yS==+1);
                    ixn = find(yS==-1);
                    
                    % Stratified splits
                    if nF*2==NS
                        foldsp = 1:nF;
                        foldsn = 1:nF;
                    else
                        foldsp = randsample(1:nF, length(ixp), true);
                        foldsn = randsample(1:nF, length(ixn), true);
                    end
                    
                    for f = 1:nF
                        
                        X = [S(ixp(f~=foldsp),:); S(ixn(f~=foldsn),:)];
                        yX = [ones(sum(f~=foldsp),1); -ones(sum(f~=foldsn),1)];
                        
                        V = [S(ixp(f==foldsp),:); S(ixn(f==foldsn),:)];
                        yV = [ones(sum(f==foldsp),1); -ones(sum(f==foldsn),1)];
                        
                        % Kernel least-squares:
                        if strcmp(ktype, 'lin')
                            theta = (X'*X + Lambda(l)*eye(2))\(X'*yX);
                            lV(s,[f==foldsp f==foldsn]') = (V*theta - yV).^2;
                            lZ(s,(f-1)*NZ+1:f*NZ) = (Z*theta - yZ).^2;
                        else
                            theta = (K(X,X,nu(b)) + Lambda(l)*eye(size(X,1)))\yX;
                            lV(s,[f==foldsp f==foldsn]') = (K(V,X,nu(b))*theta - yV).^2;
                            lZ(s,(f-1)*NZ+1:f*NZ) = (K(Z,X,nu(b))*theta - yZ).^2;
                        end
                        
                        % Retain weight
                        iwf(s,[f==foldsp f==foldsn]') = iw(d,[ixp(f==foldsp);ixn(f==foldsn)],r);
                        
                    end
                end
                
                % Analytic regression estimator (beta = Cov(lw,w)/Vw)
                if (delta(d)==1 || delta(d)>=2)
                    beta = 0;
                else
                    Cov_lw = (-4* delta(d)* sqrt(1 + delta(d))* sqrt(2* pi)* (1 + theta(2)^2) + ...
                        3* delta(d)^3* sqrt(1 + delta(d))* sqrt(2* pi)* (1 + theta(2)^2) + ...
                        delta(d)^5* theta(1)* (4 - sqrt(1 + delta(d))* sqrt(2 *pi)* theta(1)) - ...
                        4* delta(d)^2* theta(1)* (-4 + sqrt(1 + delta(d))* sqrt(2* pi)* theta(1)) - ...
                        delta(d)^4 *(sqrt(1 + delta(d))* sqrt(2* pi) + ...
                        sqrt(1 + delta(d))* sqrt(2* pi)* theta(2)^2 + 12* theta(1) - ...
                        3* sqrt(1 + delta(d))* sqrt(2* pi)* theta(1)^2) + ...
                        sqrt(2)* (-2 *(2 *sqrt(delta(d)^3 *(1 + delta(d))) + ...
                        sqrt(delta(d)^5* (1 + delta(d))) - ...
                        sqrt(delta(d)^7* (1 + delta(d))))* theta(1) + ...
                        sqrt(pi)* (4 *sqrt(-((delta(d)* (1 + delta(d)))/(-2 + delta(d)))) - ...
                        3* sqrt(-((delta(d)^5 *(1 + delta(d)))/(-2 + delta(d)))) + ...
                        sqrt(-((delta(d)^7* (1 + delta(d)))/(-2 + delta(d)))) + ...
                        (4 *sqrt(-((delta(d) *(1 + delta(d)))/(-2 + delta(d)))) - ...
                        3* sqrt(-((delta(d)^5 *(1 + delta(d)))/(-2 + delta(d)))) + ...
                        sqrt(-((delta(d)^7 *(1 + delta(d)))/(-2 + delta(d))))) *theta(2)^2 + ...
                        (2* sqrt(-((delta(d)^3 *(1 + delta(d)))/(-2 + delta(d)))) + ...
                        sqrt(-((delta(d)^5 *(1 + delta(d)))/(-2 + delta(d)))) - ...
                        sqrt(-((delta(d)^7 *(1 + delta(d)))/(-2 + delta(d)))))* theta(1)^2)))/ ...
                        ((-2 + delta(d))^2 *delta(d) *(1 + delta(d))^(3/2) *sqrt(2 *pi));
                    V_w = 1./sqrt(-delta(d).*(delta(d) - 2)) - 1;
                    beta = Cov_lw ./ V_w;
                    
                end
                if isnan(beta); beta = 0; end
                
                betah = mean((lV(:).*iwf(:) - mean(lV(:).*iwf(:),1)).*(iwf(:) - mean(iwf(:))),1) ./ ...
                    max(realmin,mean((iwf(:) - mean(iwf(:))).^2,1));
                
                MSE.V(l,b,d,r) = mean(lV(:),1);
                MSE.W(l,b,d,r) = mean(lV(:).*iwf(:),1);
                MSE.C(l,b,d,r) = mean(lV(:).*iwf(:) - beta.*(iwf(:)-1),1);
                MSE.B(l,b,d,r) = mean(lV(:).*iwf(:) - betah.*(iwf(:)-1),1);
                MSE.Z(l,b,d,r) = mean(lZ(:),1);
                
                sVh.V(l,b,d,r) = var(lV(:),[],1);
                sVh.W(l,b,d,r) = var(lV(:).*iwf(:),[],1);
                sVh.C(l,b,d,r) = var(lV(:).*iwf(:) - beta.*(iwf(:)-1),[],1);
                sVh.B(l,b,d,r) = var(lV(:).*iwf(:) - betah.*(iwf(:)-1),[],1);
                sVh.Z(l,b,d,r) = var(lZ(:),[],1);
                
            end
        end
        
        % Find minima
        [minl.V(d,r),minb.V(d,r)] = find(MSE.V(:,:,d,r) == min(reshape(MSE.V(:,:,d,r), [nL*nN 1])));
        [minl.W(d,r),minb.W(d,r)] = find(MSE.W(:,:,d,r) == min(reshape(MSE.W(:,:,d,r), [nL*nN 1])));
        [minl.C(d,r),minb.C(d,r)] = find(MSE.C(:,:,d,r) == min(reshape(MSE.C(:,:,d,r), [nL*nN 1])));
        [minl.B(d,r),minb.B(d,r)] = find(MSE.B(:,:,d,r) == min(reshape(MSE.B(:,:,d,r), [nL*nN 1])));
        [minl.Z(d,r),minb.Z(d,r)] = find(MSE.Z(:,:,d,r) == min(reshape(MSE.Z(:,:,d,r), [nL*nN 1])));
        
        %%% Optimal target risk of classifier for chosen lambdas
        if strcmp(ktype, 'lin')
            eta = @(b,l) (S'*S + Lambda(l)*eye(2))\(S'*yS);
            RT = @(eta,b) mean((Z*eta - yZ).^2,1);
        else
            eta = @(b,l) (K(S,S,nu(b)) + Lambda(l)*eye(NS))\yS;
            RT = @(eta,b) mean((K(Z,S,nu(b))*eta - yZ).^2,1);
        end
        
        % Source validation
        eta_V = eta(minb.V(d,r),minl.V(d,r));
        RTh.V(d,r) = RT(eta_V,minb.V(d,r));
        
        % Importance-weighted source validation
        eta_W = eta(minb.W(d,r),minl.W(d,r));
        RTh.W(d,r) = RT(eta_W,minb.W(d,r));
        
        % Beta controlled importance-weighted source validation
        eta_C = eta(minb.C(d,r),minl.C(d,r));
        RTh.C(d,r) = RT(eta_C,minb.C(d,r));
        
        % Beta_hat controlled importance-weighted source validation
        eta_B = eta(minb.B(d,r),minl.B(d,r));
        RTh.B(d,r) = RT(eta_B,minb.B(d,r));
        
        % Target validation
        eta_Z = eta(minb.Z(d,r),minl.Z(d,r));
        RTh.Z(d,r) = RT(eta_Z,minb.Z(d,r));
    end
    
    if viz
        fg5 = figure(5);
        clf(fg5);
        viz_MSE(MSE,Lambda, 'tix', d, 'yix', b, 'xlabel', '$$\lambda$$', 'fh', fg5)
        set(gca, 'YLim', [0 2]);
        
        fg6 = figure(6);
        clf(fg6);
        viz_RT(RTh, 'fh', fg6, 'delta', d, 'nR', nR);
        set(gca, 'YLim', [0 .8]);
    end
end


di = 1; while exist([savnm '1DG_split_iw-' iwT '_' shifttype 'shift_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm '1DG_split_iw-' iwT '_' shifttype 'shift_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'shifttype', 'delta', 'nu', 'Lambda', ...
    'iw', 'nN', 'nL', 'nR', 'nF', 'nS', 'NS','NZ', ...
    'minl', 'minb', 'MSE', 'RTh', 'sVh');

if viz
    
    figure();
    subplot(1,2,1);
    hist(dVZ(:));
    subplot(1,2,2);
    hist(dVS(:));
    
end
    
