function X = sampleDist2D(f,M,N,b1,b2)
% Adapted from sampleDist (Dmitry Savransky, dsavrans@princeton.edu, May, 2010)
%
%     sampleDist2D(f,M,N,b1,b2) retruns an array of size X of random values
%     sampled from the distribution defined by the probability density
%     function f, over the range b1 = [min, max] and b2 = [min, max]. 
%     M is the threshold value for the proposal distribution, such that 
%     f(x) < M for all x in b.
%
% Wouter Kouw, 2016

% Initialize
n = 0;
c = 0;

% Preallocate
X = NaN(N,2);

while n < N
    
    % Generate grid uniform random values
    x = bsxfun(@plus,[b1(1) b2(1)], bsxfun(@times, rand(2*N,2), [diff(b1) diff(b2)]));
    
    % Generate proposal values
    uM = M*rand(2*N,1);
    
    % Accept samples
    x = x(uM < f(x(:,1),x(:,2)),:);
    
    % Number of accepted samples
    nA = size(x,1);
    
    % Add to existing set
    X(n+1:min([n+nA,N]),:) = x(1:min([nA,N - n]),:);
    
    % Tick up
    n = n + nA;    
    c = c+1;
    
    % Check for cycling
    if c > 1e4
        error('too many iterations');
    end
end

