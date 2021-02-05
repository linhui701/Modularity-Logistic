function Y = getNormalization(X)
% --------------------------------------------------------------------
% Normalizating data set
% --------------------------------------------------------------------

[~,n] = size(X);
Y = X;

for j = 1 : n
    Xv = X(:,j);
    Xvn = (Xv-mean(Xv))/std(Xv);
    Y(:,j) = Xvn;
end