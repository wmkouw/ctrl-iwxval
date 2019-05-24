function [iw,theta] = iw_KMM(X,Z,varargin)
% Use Kernel Mean Matching to estimate weights for importance weighting.
%
% Jiayuan Huang, Alex Smola, Arthur Gretton, Karsten Borgwardt & Bernhard
% Schoelkopf. Correcting Sample Selection Bias by unlabeled data.

% Parse optionals
p = inputParser;
addOptional(p, 'theta', 0);
addOptional(p, 'eps', 0);
addOptional(p, 'B', 1e3);
addOptional(p, 'nF', 2);
addOptional(p, 'off', 0);
addOptional(p, 'clip', Inf);
parse(p, varargin{:});

% Shapes
[N,D] = size(X);
[M,~] = size(Z);

% Optimization options
options.Display = 'valid';
% options.Algorithm = 'active-set';
options.TolCon = 1e-5;

% Constraints
if p.Results.eps==0
    eps = [p.Results.B./sqrt(N) p.Results.B./sqrt(N)];
elseif length(p.Results.eps)==2
    eps = p.Results.eps;
else
    eps = [p.Results.eps p.Results.eps];
end
A = [ones(1,N); -ones(1,N)];
b = [N*(eps(1)+1); N*(eps(2)-1)]+p.Results.off;
lb = zeros(N,1);
ub = p.Results.B*ones(N,1);

% Optimal bandwidth crossvalidation
if p.Results.theta==0
    
    % Maximum Mean Discrepancy
    MMD = @(beta,K,k) 1./size(K,1)^2*beta'*K*beta - 2./size(K,1)^2*k'*beta;
    
    theta = 10; 
    score = -Inf;
    for epsilon = log10(theta)-1:-1:-5
        
        for iteration=1:9
            
            % Update sigma
            sigma_new = theta-10^epsilon;
            
            permM = randperm(M);
            cvsplit = floor([0:M-1]*p.Results.nF./M)+1;
            
            % Compute kernels
            KXX = exp(-pdist2(X,X)./sigma_new);
            KXZ = exp(-pdist2(X,Z)./sigma_new);
            
            score_new = 0;
            for f = 1:p.Results.nF
                
                % Find beta's using given target samples
                knf = N./(sum(cvsplit~=f)).*sum(KXZ(:,permM(cvsplit~=f)),2);
                beta_cv = quadprog(KXX,knf,A,b,[],[],lb, ub, [], options);
                
                % Evaluate MMD for beta's on held-out target samples
                khf = N./(sum(cvsplit==f)).*sum(KXZ(:,permM(cvsplit==f)),2);
                score_new = score_new + MMD(beta_cv,KXX,khf);
            end
            
            if (score_new - score) > 0
                break
            end
            score = score_new;
            theta = sigma_new;
            fprintf('  score=%g,  sigma=%g\n',score,theta)
        end
    end
    fprintf('Final sigma = %g\n',theta)
else
    theta = p.Results.theta;
end

% Compute kernels with optimized theta
KXX = exp(-pdist2(X,X)./theta);
KXZ = exp(-pdist2(X,Z)./theta);

% Optimize
iw = quadprog(KXX,mean(KXZ,2),A,b,[],[],lb, ub, [], options);

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

end
