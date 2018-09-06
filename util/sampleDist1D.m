function X = sampleDist1D(f,M,N,b)
% Adapted from sampleDist (Dmitry Savransky, dsavrans@princeton.edu, May, 2010)
%
%     sampleDist1D(f,M,N,b) retruns an array of size X of random values
%     sampled from the distribution defined by the probability density
%     function f, over the range b = [min, max]. 
%     M is the threshold value for the proposal distribution, such that 
%     f(x) < M for all x in b.
%
% Wouter Kouw, 2016

% Initialize
n = 0;
c = 0;

% Preallocate
X = NaN(N,1);

while n < N

    % Generate grid uniform random values
    x = b(1) + rand(2*N,1)*diff(b);

    % Generate proposal values
    uM = M*rand(2*N,1);

    % Accept samples
    x = x(uM < f(x));

    % Number of accepted samples
    nA = size(x,1);

    % Add to existing set
    X(n+1:min([n+length(x),N])) = x(1:min([length(x),N - n]));

    % Tick up
    n = n + nA;
    c = c+1;

    % Check for cycling
    if c > 1e4
        error('too many iterations');
    end
end

