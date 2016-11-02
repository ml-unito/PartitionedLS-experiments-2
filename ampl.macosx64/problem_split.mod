param M;
param N;
param X{1..N,1..M};
param K;
param y{1..N};
param beta{1..K};
param h{1..(K+1)} in {1..(M+1)};
var alpha{1..M}>=0;

minimize ab: sum{n in  1..N}(sum{k in 1..K}beta[k]*(sum{m in h[k]..(h[k+1]-1)} alpha[m]*X[n,m])-y[n])^2;
