function [iw] = iw_Gauss(X, Z, varargin)
% Uses two Gaussian distributions to estimate importance weights
% ! Do not use for high-dimensional data
% Function expects NxD matrices.

% Parse optionals
p = inputParser;
addOptional(p, 'l2', realmin);
addOptional(p, 'clip', realmax);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Shape
[~,D] = size(X);

% Compute probability under target Gaussian distribution
pZ_X = mvnpdf(X, mean(Z,1), cov(Z)+p.Results.l2*eye(D));

% Compute probability under source Gaussian distribution
pX_X = mvnpdf(X, mean(X,1), cov(X)+p.Results.l2*eye(D));

% Compute ratio of probabilities
iw = pZ_X ./ pX_X;

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

if p.Results.viz
    figure()
    histogram(iw);
end

end
