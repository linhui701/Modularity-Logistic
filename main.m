clc;clear;
% ----------------------------------------------------------------------- %
% Run this example to see how to use
% ----------------------------------------------------------------------- %
% Author: Linhui Xie, linhxie@iu.edu, xie215@purdue.edu
% Date created: Jan-11-2020
% Update: Feb-05-2021
% @Indiana University School of Medicine.
% @Purdue University Electrical and Computer Engineering.
% ----------------------------------------------------------------------- %

%% load ROSMAP data sets
load('Dataset_M_Logistic.mat')
% contain data matrix x_adjust, disease status vector y_clinical.
% contain modularity matrix B_max_abs_eig.

x=x_adjust;
y=y_clinical;

%% remove the subjects with no snps data
[n_sub, n_feature] = size(x);

no_snps = zeros(1, n_feature-929); % Non-SNPs feature 1~929
id_snps = ones(n_sub, 1);

for i = 1:n_sub
    if x(i,930:end) == no_snps
        id_snps(i)=0;
    end
end
y = y(id_snps==1);
x = x(id_snps==1, :);

%% build X and Y matrix
% Three categories, 1(HC), 2(MCI) and 4(AD)
X = x;
Y = zeros(length(y),1);
Y(y==1) = 0;
Y(y==2) = 0;
Y(y==4) = 1;
Y = logical(Y);

%% Replace the NaN value in X and Y matrix
for i = 1:929
    [row,col]=find(isnan(X(:,i))==1);
    col_mean = nanmean(X(:,i));
    for j = 1:length(row)
       X(row(j),i) =  col_mean;
    end
end

%% Normalize matrix X
X = getNormalization(X);


%% RUN LOGISTICMODULAR
% Set the seed for reproducibility
rng(600)
c = cvpartition(Y,'KFold',10,'Stratify',true);   % Keep the AD:HC ratio

alpha = 0.1;
lambda_list = [1*10^-6, 1*10^-5, 1*10^-4, 1*10^-3, 1*10^-2, 1*10^-1];

for i = 1 : 10
    XTrain = X(training(c,i), :);
    yTrain = Y(training(c,i));
    XTest  = X(~training(c,i), :);
    yTest  = Y(~training(c,i));
    
    % for reproducibility
    rng('default')  
    
    [W, FitInfo] = logisticmodular( XTrain, yTrain, B_max_abs_eig, ...
                                    alpha, lambda_list, ...
                                    'binomial', 'CV', 3);
    lambda = FitInfo.LambdaMinDeviance;
    index  = FitInfo.IndexMinDeviance;
    dev    = FitInfo.Deviance(index);
    W0 = FitInfo.Intercept;
        
    % testing
    W_B = [W0; W];
    yTest_hat = glmval(W_B, XTest, 'logit');
    yTest_hatBinom = (yTest_hat>0.5);
    cTest = confusionchart(yTest, yTest_hatBinom);
    cmTest = cTest.NormalizedValues;
    accTest = (cmTest(1,1) + cmTest(2,2)) / sum(cmTest(:));
    fprintf('%dth fold result accuracy: %.4f\n', i, accTest)

    save(sprintf('fold_%d.mat', i), 'W_B', 'accTest', 'dev');
end

