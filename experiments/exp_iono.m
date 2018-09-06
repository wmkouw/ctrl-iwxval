% Script to run some covariate shift experiments
close all;
clearvars;

addpath(genpath('../util'))

sav = true;
viz = false;
savnm = 'results/';

%% Load pima dataset

load iono_remf2
y(y==2) = -1;
py = [mean(y==-1) mean(y==1)];

% D = zscore(D,[],1);
% D = bsxfun(@minus, D, mean(D,1));
[~,C,~] = pca(D);
% C = zscore(C,[],1);

% C = C*5;

% Reduce dimensionality to 2
C = C(:,1:2);

% Split by class
ixn = find(y==-1);
ixp = find(y==+1);
Cn = C(ixn,:);
Cp = C(ixp,:);

% Sizes
N = size(C,1);
Nn = size(Cn,1);
Np = size(Cp,1);

% Data range limits
u1l = [min(C(:,1),[],1) max(C(:,1),[],1)];
u2l = [min(C(:,2),[],1) max(C(:,2),[],1)];

%%

% Number of folds
nF = 2;

% Number of times split
nS = 1;

% Number of repeats
nR = 10;

% Sample sizes
NS = 40;   % source data
NZ = N-NS;  % target training

% Importance weight estimator
iwT = 'gauss';

% Means
mu_S = zeros(1,size(C,2));
mu_T = zeros(1,size(C,2));

% Target parameter changes
shifttype = 'var';
% gamma2 = 2.^[-4:1:4];
gamma2 = 0.5^2;
nD = length(gamma2);

% Quadratic basis function
ktype = 'pol';
switch ktype
    case 'pol'
        K = @(x,y,d) (x*y'+1).^d;
    case 'rbf1'
        K = @(x,y,d) exp(-pdist2(x,y)./d);
    case 'rbf2'
        K = @(x,y,l) exp(- pdist2(x,y,'Mahalanobis', diag(l)));
end

% Hyperparameter ranges
nL = 100;
switch ktype
    case {'rbf1','rbf2'}
        Lambda = linspace(0.01,1,nL);
        if C <= 2
            [~,~,nu] = ksdensity(C);
        else
            nu = 0.5;
        end
        nN = size(nu,1);
    otherwise
        Lambda = linspace(0.01,10,nL)*NS;
        nu = 2;
        nN = 1;
end

fS = 15;
mS = 40;

%%

iw = NaN(nD,NS,nR);
% beta = NaN(nL,nN,nD,nR);
MSE.V = NaN(nL,nN,nD,nR);
MSE.W = NaN(nL,nN,nD,nR);
MSE.B = NaN(nL,nN,nD,nR);
MSE.Z = NaN(nL,nN,nD,nR);
sVh.V = NaN(nL,nN,nD,nR);
sVh.W = NaN(nL,nN,nD,nR);
sVh.B = NaN(nL,nN,nD,nR);
sVh.Z = NaN(nL,nN,nD,nR);
minl.V = NaN(nD,nR);
minl.W = NaN(nD,nR);
minl.B = NaN(nD,nR);
minl.Z = NaN(nD,nR);
minb.V = NaN(nD,nR);
minb.W = NaN(nD,nR);
minb.B = NaN(nD,nR);
minb.Z = NaN(nD,nR);
RTh.V = NaN(nD,nR);
RTh.W = NaN(nD,nR);
RTh.B = NaN(nD,nR);
RTh.Z = NaN(nD,nR);
eTh.V = NaN(nD,nR);
eTh.W = NaN(nD,nR);
eTh.B = NaN(nD,nR);
eTh.Z = NaN(nD,nR);

for d = 1:nD
    disp(['Change in parameter: ' num2str(gamma2(d))]);
    
    % Current set of gamma
    Sigma_S = gamma2(d)*cov(C);
    
    % Sampling probabilities
    pS = mvnpdf(C, mu_S, Sigma_S);
    
    for r = 1:nR
        
        % Sample source data
        [Sn,ixSn] = datasample(Cn,round(py(1)*NS), 'Replace', false, 'Weights', pS(ixn));
        [Sp,ixSp] = datasample(Cp,round(py(2)*NS), 'Replace', false, 'Weights', pS(ixp));
        
        % Take set difference with source as target
        Zn = Cn(setdiff(1:Nn,ixSn),:);
        Zp = Cp(setdiff(1:Np,ixSp),:);
        
        % Concatenate to datasets
        Z = [Zn; Zp];
        S = [Sn; Sp];
        yZ = [-ones(size(Zn,1),1); ones(size(Zp,1),1)];
        yS = [-ones(size(Sn,1),1); ones(size(Sp,1),1)];% Obtain importance weights
        
        if viz
            
            xl = [min(C(:,1))-0.5 max(C(:,1))+0.5];
            yl = [min(C(:,2))-0.5 max(C(:,2))+0.5];
            xr = linspace(xl(1),xl(2),101);
            yr = linspace(yl(1),yl(2),101);
            [tx,ty] = meshgrid(xr,yr);
            
            fg1 = figure(1);
            hold on
            scatter(Cn(:,1),Cn(:,2), mS-30, 'filled', 'ko');
            scatter(Cp(:,1),Cp(:,2), mS-30, 'filled', 'ko');
            scatter(Sn(:,1),Sn(:,2), mS, 'filled', 'rs');
            scatter(Sp(:,1),Sp(:,2), mS, 'filled', 'bs');
            contour(xr',yr',reshape(mvnpdf([tx(:) ty(:)], mu_S, Sigma_S), [101 101]), 'LineColor', 'k', 'DisplayName', 'p_S')
            xlabel('$$x_1$$', 'Interpreter', 'latex');
            ylabel('$$x_2$$', 'Interpreter', 'latex');
            gt = findobj(gcf, 'Type', 'Scatter');
            legend(gt(1:2), {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex');
            set(gca, 'XLim', xl, 'YLim', yl, 'FontSize', fS);
            set(gcf, 'Color', 'w', 'Position', [100 100 1000 600]);
            
            if sav; saveas(gcf, 'viz/iono_S.eps', 'epsc'); end
            
            fg2 = figure(2);
            hold on
            scatter(Cn(:,1),Cn(:,2), mS-20, 'filled', 'ko');
            scatter(Cp(:,1),Cp(:,2), mS-20, 'filled', 'ko');
            scatter(Zn(:,1),Zn(:,2), mS, 'filled', 'md');
            scatter(Zp(:,1),Zp(:,2), mS, 'filled', 'cd');
            %             contour(xr',yr',reshape(mvnpdf([tx(:) ty(:)], mu_T, Sigma_T), [101 101]), 'LineColor', 'k', 'DisplayName', 'p_T');
            xlabel('$$x_1$$', 'Interpreter', 'latex');
            ylabel('$$x_2$$', 'Interpreter', 'latex');
            gt = findobj(gcf, 'Type', 'Scatter');
            legend(gt(1:2), {'$$y= -1$$', '$$y=+1$$'}, 'Interpreter', 'latex');
            set(gca, 'XLim', xl, 'YLim', yl, 'FontSize', fS);
            set(gcf, 'Color', 'w', 'Position', [100 100 1000 600]);
            
            if sav; saveas(gcf, 'viz/iono_T.eps', 'epsc'); end
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
                    iSp = find(yS==+1);
                    iSn = find(yS==-1);
                    
                    % Stratified splits
                    foldsp = randsample(1:nF, length(iSp), true);
                    foldsn = randsample(1:nF, length(iSn), true);
                    
                    for f = 1:nF
                        
                        X = [S(iSp(f~=foldsp),:); S(iSn(f~=foldsn),:)];
                        yX = [ones(sum(f~=foldsp),1); -ones(sum(f~=foldsn),1)];
                        
                        V = [S(iSp(f==foldsp),:); S(iSn(f==foldsn),:)];
                        yV = [ones(sum(f==foldsp),1); -ones(sum(f==foldsn),1)];
                        
                        % Kernel least-squares:
                        theta = (K(X,X,nu(b,:)) + Lambda(l)*eye(size(X,1)))\yX;
                        
                        % Compute pointwise loss
                        lV(s,[f==foldsp f==foldsn]') = (K(V,X,nu(b,:))*theta - yV).^2;
                        lZ(s,(f-1)*NZ+1:f*NZ) = (K(Z,X,nu(b,:))*theta - yZ).^2;
                        
                        switch lower(iwT)
                            case 'none'
                                iwf(s,[f==foldsp f==foldsn]') = ones(1,nV);
                            case 'gauss'
                                iwf(s,[f==foldsp f==foldsn]') = iw_Gauss(V,Z, 'lambda', 1e-5);
                            case 'kde'
                                iwf(s,[f==foldsp f==foldsn]') = iw_kde(V,Z, 'bw', []);
                            case 'kmm'
                                iwf(s,[f==foldsp f==foldsn]') = iw_KMM(V,Z, 'theta', 1);
                            case 'kliep'
                                iwf(s,[f==foldsp f==foldsn]') = iw_KLIEP(V,Z,0,realmax);
                            case 'nnew'
                                iwf(s,[f==foldsp f==foldsn]') = iw_NNeW(V,Z,0,realmax, 'Laplace', 1);
                            otherwise
                                error('Unknown importance weight estimator');
                        end
%                         
%                         % Retain weight
%                          iwf(s,[f==foldsp f==foldsn]') = iw(d,[iSp(f==foldsp);iSn(f==foldsn)],r);
                        
                    end
                end
                
                beta = mean((lV(:).*iwf(:) - mean(lV(:).*iwf(:),1)).*(iwf(:) - 1),1) ./ ...
                    max(realmin,mean((iwf(:) - 1).^2,1));
                
                MSE.V(l,b,d,r) = mean(lV(:),1);
                MSE.W(l,b,d,r) = mean(lV(:).*iwf(:),1);
                MSE.B(l,b,d,r) = mean(lV(:).*iwf(:) - beta.*(iwf(:)-1),1);
                MSE.Z(l,b,d,r) = mean(lZ(:),1);
                
                sVh.V(l,b,d,r) = var(lV(:),[],1);
                sVh.W(l,b,d,r) = var(lV(:).*iwf(:),[],1);
                sVh.B(l,b,d,r) = var(lV(:).*iwf(:) - beta.*(iwf(:)-1),[],1);
                sVh.Z(l,b,d,r) = var(lZ(:),[],1);
                
            end
        end
        
        % Find minima
        [minl.V(d,r),minb.V(d,r)] = find(MSE.V(:,:,d,r) == min(reshape(MSE.V(:,:,d,r), [nL*nN 1])), 1);
        [minl.W(d,r),minb.W(d,r)] = find(MSE.W(:,:,d,r) == min(reshape(MSE.W(:,:,d,r), [nL*nN 1])), 1);
        [minl.B(d,r),minb.B(d,r)] = find(MSE.B(:,:,d,r) == min(reshape(MSE.B(:,:,d,r), [nL*nN 1])), 1);
        [minl.Z(d,r),minb.Z(d,r)] = find(MSE.Z(:,:,d,r) == min(reshape(MSE.Z(:,:,d,r), [nL*nN 1])), 1);
        
        %%% Optimal target risk of classifier for chosen lambdas
        eta = @(b,l) (K(S,S,nu(b,:)) + Lambda(l)*eye(NS))\yS;
        RT = @(eta,b) mean((K(Z,S,nu(b,:))*eta - yZ).^2,1);
        eT = @(eta,b) mean(sign(K(Z,S,nu(b,:))*eta) ~= yZ,1);
        
        % Source validation
        eta_V = eta(minb.V(d,r),minl.V(d,r));
        RTh.V(d,r) = RT(eta_V,minb.V(d,r));
        eTh.V(d,r) = eT(eta_V,minb.V(d,r));
        
        % Importance-weighted source validation
        eta_W = eta(minb.W(d,r),minl.W(d,r));
        RTh.W(d,r) = RT(eta_W,minb.W(d,r));
        eTh.W(d,r) = eT(eta_W,minb.W(d,r));
        
        % Beta controlled importance-weighted source validation
        eta_B = eta(minb.B(d,r),minl.B(d,r));
        RTh.B(d,r) = RT(eta_B,minb.B(d,r));
        eTh.B(d,r) = eT(eta_B,minb.B(d,r));
        
        % Target validation
        eta_Z = eta(minb.Z(d,r),minl.Z(d,r));
        RTh.Z(d,r) = RT(eta_Z,minb.Z(d,r));
        eTh.Z(d,r) = eT(eta_Z,minb.Z(d,r));
    end
end


di = 1; while exist([savnm 'iono_uniftgt_iw-' iwT '_Ktype' ktype '_' shifttype 'shift_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm 'iono_uniftgt_iw-' iwT '_Ktype' ktype '_' shifttype 'shift_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'shifttype', 'gamma2', 'nu', 'Lambda', ...
    'iw', 'nN', 'nL', 'nR', 'nF', 'nS', 'NS','NZ', ...
    'minl', 'minb', 'MSE', 'RTh', 'eTh', 'sVh');
