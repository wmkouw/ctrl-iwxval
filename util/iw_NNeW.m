function [iw] = iw_NNeW(X,Z,varargin)
% Use Nearest-Neighbour to estimate weights for importance weighting.
%
% Marco Loog. Nearest Neighbour-Based Importance Weighting. MLSP 2012

% Parse optionals
p = inputParser;
addOptional(p, 'clip', Inf);
addOptional(p, 'Laplace', false);
parse(p, varargin{:});

% Calculate Euclidean distance
D = pdist2(X, Z);

% Count how many target samples are in Voronoi Tesselation
[~,ix] = min(D, [], 1);
iw = histcounts(ix, (1:size(X,1)+1)-0.5);

% Laplace smoothing
if p.Results.Laplace
    iw = iw + 1;
end

% Weight clipping
iw = min(p.Results.clip, max(0, iw))';

end
