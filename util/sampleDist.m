function X = sampleDist(f,M,N,b)
% SAMPLEDIST  Sample from an arbitrary distribution
%     sampleDist(f,M,N,b) retruns an array of size X of random values
%     sampled from the distribution defined by the probability density
%     function refered to by handle f, over the range b = [min, max].
%     M is the threshold value for the proposal distribution, such that
%     f(x) < M for all x in b.
%
%     sampleDist(...,true) also generates a histogram of the results
%     with an overlay of the true pdf.
%
%     Examples:
%     %Sample from a step function over [0,1]:
%     X = sampleDist(@(x)1.3*(x>=0&x<0.7)+0.3*(x>=0.7&x<=1),...
%                    1.3,1e6,[0,1],true);
%     %Sample from a normal distribution over [-5,5]:
%     X = sampleDist(@(x) 1/sqrt(2*pi) *exp(-x.^2/2),...
%                    1/sqrt(2*pi),1e6,[-5,5],true);
%

% Dmitry Savransky (dsavrans@princeton.edu)
% May 11, 2010

n = 0;
X = zeros(N,1);
counter = 0;

while 1
    while n < N && counter < 1e6
        x = b(1) + rand(2*N,1)*diff(b);
        uM = M*rand(2*N,1);
        x = x(uM < f(x));
        X(n+1:min([n+length(x),N])) = x(1:min([length(x),N - n]));
        n = n + length(x);
        counter = counter+1;
    end
    if ~isempty(x)
        break;
    end
end

