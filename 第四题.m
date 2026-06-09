% Population Coding
% Poisson Neurons
% MVUB Estimation
% Fisher Information Verification

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

clear;
clc;
close all;

rng(1);

%% Parameters

N = 20;                  % number of neurons

T = 100;                 % observation window

s_true = 90;             % true stimulus direction

sigma_tune = 20;         % tuning width

%% Preferred directions

s_pref = linspace(0,180,N+1);
s_pref(end) = [];

sigma = sigma_tune*ones(1,N);

%% Tuning curves

f = exp(-(s_true-s_pref).^2./(2*sigma.^2));

lambda = T*f;

%% Plot tuning curves

s_grid = linspace(0,180,1000);

figure

hold on

for a=1:N

    fa = exp(-(s_grid-s_pref(a)).^2/(2*sigma(a)^2));

    plot(s_grid,fa)

end

xlabel('Stimulus Direction')

ylabel('Firing Rate')

title('Neuronal Tuning Curves')

%% Fisher Information

Fisher = 0;

for a=1:N

    term = ...
        ((s_true-s_pref(a))^2/sigma(a)^4) ...
        *exp(-(s_true-s_pref(a))^2/(2*sigma(a)^2));

    Fisher = Fisher + term;

end

Fisher = T*Fisher;

CRLB = 1/Fisher;

fprintf('\n');
fprintf('========================\n');
fprintf('Fisher Information\n');
fprintf('========================\n');

fprintf('Fisher = %.6f\n',Fisher);

fprintf('CRLB   = %.6f\n',CRLB);

%% Monte Carlo

MC = 5000;

s_hat = zeros(MC,1);

%% MLE estimation

s_search = linspace(0,180,4001);

for mc=1:MC

    % Generate spike counts

    Nspike = poissrnd(lambda);

    % Log likelihood

    LogLike = zeros(size(s_search));

    for k=1:length(s_search)

        s = s_search(k);

        ftemp = ...
            exp(-(s-s_pref).^2./(2*sigma.^2));

        lam = T*ftemp;

        LogLike(k) = ...
            sum( Nspike.*log(lam+eps) ...
            - lam );

    end

    % MLE

    [~,idx] = max(LogLike);

    s_hat(mc) = s_search(idx);

end

%% Statistics

MeanHat = mean(s_hat);

VarHat = var(s_hat);

MSE = mean((s_hat-s_true).^2);

fprintf('\n');
fprintf('========================\n');
fprintf('Monte Carlo Results\n');
fprintf('========================\n');

fprintf('True Direction     = %.4f\n',s_true);

fprintf('Mean Estimate      = %.4f\n',MeanHat);

fprintf('Variance Estimate  = %.6f\n',VarHat);

fprintf('MSE                = %.6f\n',MSE);

fprintf('CRLB               = %.6f\n',CRLB);

%% Histogram of estimator

figure

histogram(s_hat,50,...
'Normalization','pdf')

hold on

x = linspace(min(s_hat),max(s_hat),1000);

y = normpdf(x, s_true, sqrt(CRLB));

plot(x,y,...
'LineWidth',2)

xlabel('$\\hat{s}$')

ylabel('Density')

title('Distribution of MLE')

%% Compare variance and CRLB

figure

bar([VarHat CRLB])

set(gca,...
'XTickLabel',...
{'Simulation','CRLB'})

ylabel('Variance')

title('Variance vs Cramer-Rao Bound')

%% MSE convergence

MSE_curve = zeros(MC,1);

for k=1:MC

    MSE_curve(k) = ...
        mean((s_hat(1:k)-s_true).^2);

end

figure

plot(MSE_curve,...
'LineWidth',2)

hold on

yline(CRLB,...
'r--',...
'LineWidth',2)

xlabel('Monte Carlo Samples')

ylabel('MSE')

legend('Simulation','CRLB')

title('MSE Convergence')

%% Fisher information as function of s

FisherCurve = zeros(size(s_grid));

for k=1:length(s_grid)

    s = s_grid(k);

    F = 0;

    for a=1:N

        F = F + ...
        ((s-s_pref(a))^2/sigma(a)^4) ...
        *exp(-(s-s_pref(a))^2/(2*sigma(a)^2));

    end

    FisherCurve(k)=T*F;

end

figure

plot(s_grid,...
FisherCurve,...
'LineWidth',2)

xlabel('Direction')

ylabel('Fisher Information')

title('Population Fisher Information')

%% End

disp('Simulation Finished.')