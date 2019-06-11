function [iw] = iw_KLIEP(X,Z,varargin)
% Kullback-Leibler Importance Estimation Procedure
%
% Sugiyama, Nakajima, Kashima, von Bunau & Kawanabe. Direct Importance
% Estimation with Model Selection and Its Application to Covariate Shift
% (NIPS, 2008).

% Parse optionals
p = inputParser;
addOptional(p, 'nK',100);
addOptional(p, 'sigma', 0);
addOptional(p, 'fold',5);
addOptional(p, 'clip', Inf);
addOptional(p, 'verbose', false);
parse(p, varargin{:});

% Shape
[N, D]=size(X);
[M, ~]=size(Z);

% Center kernel on nK random target points
rand_index = randperm(M);
nK = min(p.Results.nK, M);
x_ce = Z(rand_index(1:nK), :);

if p.Results.sigma == 0
    
    % Cross-validation for kernel bandwidth
    sigma = 10; 
    score = -inf;
    
    for epsilon = log10(sigma)-1:-1:-5
        for iteration = 1:9
            
            % Update kernel bandwidth
            sigma_new = sigma-10^epsilon;
            
            % Create split index
            cv_index = randperm(M);
            cv_split = floor([0:M-1]*p.Results.fold./M) + 1;
            score_new = 0;
            
            % Fit kernels
            X_de = kernel_Gaussian(X, x_ce, sigma_new);
            X_nu = kernel_Gaussian(Z, x_ce, sigma_new);
            mean_X_de = mean(X_de,1)';
            
            % Loop over folds
            for i = 1:p.Results.fold
                
                % Estimate alpha parameters through KLIEP
                alpha_cv = KLIEP_learning(mean_X_de, X_nu(cv_index(cv_split~=i),:));
                
                % Compute weights
                wh_cv = X_nu(cv_index(cv_split==i),:)*alpha_cv;
                
                % Estimate score
                score_new = score_new + mean(log(wh_cv)) / p.Results.fold;
            end
            
            % Break on improvement
            if (score_new-score)<=0
                break
            end
            
            % Update
            score = score_new;
            sigma = sigma_new;
            
            % Report
            if p.Results.verbose
                fprintf('  score=%g,  sigma=%g \n',score,sigma)
            end
        end
    end
else
    % Stick with chosen kernel bandwidth
    sigma = p.Results.sigma;
end

% Report
if p.Results.verbose
    fprintf('sigma = %g \n',sigma)
end

% Computing the final solution 'iw'
X_de = kernel_Gaussian(X,x_ce,sigma);
X_nu = kernel_Gaussian(Z,x_ce,sigma);
mean_X_de = mean(X_de,1)';
alphah = KLIEP_learning(mean_X_de,X_nu);
iw = X_de*alphah;

% Weight clipping
iw = min(p.Results.clip, max(0, iw));

end

function px = pdf_Gaussian(x,mu,sigma)

[d,nx] = size(x);

tmp = (x-repmat(mu,[1 nx]))./repmat(sigma,[1 nx])/sqrt(2);
px = (2*pi)^(-d/2)/prod(sigma)*exp(-sum(tmp.^2,1));

end

function X = kernel_Gaussian(x,c,sigma)

[nx, d] = size(x);
[nc, d] = size(c);

distance2 = repmat(sum(c.^2,2), [1 nx])'+repmat(sum(x.^2,2)', [nc 1])'-2*x*c';
X = exp(-distance2/(2*sigma^2));

end

function [alpha, Xte_alpha, score] = KLIEP_projection(alpha, Xte, b, c)

%  alpha=alpha+b*(1-sum(b.*alpha))/c;

alpha=alpha+b*(1-sum(b.*alpha))*pinv(c,10^(-20));

alpha=max(0,alpha);

%  alpha=alpha/sum(b.*alpha);

alpha=alpha*pinv(sum(b.*alpha),10^(-20));

Xte_alpha=Xte*alpha;

score=mean(log(Xte_alpha));

end

function [alpha,score] = KLIEP_learning(mean_X_de,X_nu)

[n_nu, nc] = size(X_nu);

max_iteration = 100;
epsilon_list = 10.^[3:-1:-3];

% Preallocate
alpha = ones(nc,1);

% Initialize
c = sum(mean_X_de.^2);
[alpha, X_nu_alpha, score] = KLIEP_projection(alpha, X_nu, mean_X_de, c);

for epsilon = epsilon_list
    for iteration = 1:max_iteration
        
        % Find new alpha's through projection
        alpha_tmp = alpha + epsilon*X_nu'*(1./X_nu_alpha);
        [alpha_new,X_nu_alpha_new,score_new] = KLIEP_projection(alpha_tmp,X_nu,mean_X_de,c);
        
        % Break on improvement
        if (score_new-score)<=0
            break
        end
        
        % Update
        score = score_new;
        alpha = alpha_new;
        X_nu_alpha = X_nu_alpha_new;
    end
end

end
