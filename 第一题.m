
%  LIF + OU Synaptic Current
%  Monte Carlo Calibration Version
%  Verify:
%     E[T]
%     Var(T)
%     H(t)=E[N(t)]
%     Var(N(t))

clear;
clc;
close all;

rng(1);

%% Model Parameters

gL      = 0.1;

Vth     = 1.0;
Vr      = 0.0;

tau_ref = 2.0;

tau_syn = 0.5;

%% Target Statistics

m_target   = 20;

var_target = 25;

%% Construct Mean

gsyn = 1;

A = exp(gL*(m_target-tau_ref));

mu = gL*(A*Vth-Vr)/(gsyn*(A-1));

fprintf('\n');
fprintf('============================\n');
fprintf('Mean Construction\n');
fprintf('============================\n');
fprintf('gsyn = %.4f\n',gsyn);
fprintf('mu   = %.4f\n',mu);

%% Monte Carlo Calibration for sigma

sigma_low  = 0.001;
sigma_high = 0.30;

for iter=1:15

    sigma = (sigma_low+sigma_high)/2;

    ISI = simulateISI( ...
        gL,gsyn,mu,sigma,...
        Vr,Vth,...
        tau_ref,tau_syn);

    vhat = var(ISI);

    fprintf('Iter=%2d sigma=%8.5f var=%8.4f\n',...
        iter,sigma,vhat);

    if vhat>var_target

        sigma_high=sigma;

    else

        sigma_low=sigma;

    end

end

sigma=(sigma_low+sigma_high)/2;

fprintf('\n');
fprintf('============================\n');
fprintf('Calibrated Parameters\n');
fprintf('============================\n');

fprintf('gsyn  = %.5f\n',gsyn);
fprintf('mu    = %.5f\n',mu);
fprintf('sigma = %.5f\n',sigma);

%% Large Simulation

dt=0.01;

Tmax=50000;

time=0:dt:Tmax;

Nt=length(time);

V=Vr;

Isyn=mu;

last_spike=-1e10;

spike_times=[];

V_trace=zeros(1,Nt);

I_trace=zeros(1,Nt);

for n=1:Nt

    t=time(n);

    if (t-last_spike)<tau_ref

        V=Vr;

    else

        dW=sqrt(dt)*randn;

        Isyn=Isyn...
            +dt/tau_syn*(-Isyn+mu)...
            +sigma/tau_syn*dW;

        V=V+dt*(-gL*V+gsyn*Isyn);

        if V>=Vth

            spike_times=[spike_times,t];

            last_spike=t;

            V=Vr;

        end

    end

    V_trace(n)=V;

    I_trace(n)=Isyn;

end

%% ISI Statistics

ISI=diff(spike_times);

m_hat=mean(ISI);

var_hat=var(ISI);

fprintf('\n');
fprintf('============================\n');
fprintf('ISI Statistics\n');
fprintf('============================\n');

fprintf('Target Mean     = %.4f\n',m_target);
fprintf('Sample Mean     = %.4f\n',m_hat);

fprintf('Target Variance = %.4f\n',var_target);
fprintf('Sample Variance = %.4f\n',var_hat);

%% Figure 1

figure;

plot(time(1:50000),I_trace(1:50000),'LineWidth',1.2);

xlabel('t');

ylabel('I_{syn}');

title('OU Synaptic Current');

%% Figure 2

figure;

plot(time(1:50000),V_trace(1:50000),'LineWidth',1.2);

xlabel('t');

ylabel('V');

title('Membrane Potential');

%% Figure 3

figure;

histogram(ISI,40,'Normalization','pdf');

xlabel('ISI');

ylabel('Density');

title('ISI Distribution');

%% Renewal Process

M=1000;

TmaxRenew=2000;

tGrid=linspace(100,TmaxRenew,100);

Ng=length(tGrid);

Nall=zeros(M,Ng);

Nsample=length(ISI);

for r=1:M

    idx=randi(Nsample,5000,1);

    Trenew=ISI(idx);

    S=cumsum(Trenew);

    for k=1:Ng

        Nall(r,k)=sum(S<=tGrid(k));

    end

end

%% Renewal Statistics

H_emp=mean(Nall);

Var_emp=var(Nall);

H_theory=tGrid/m_hat;

Var_theory=(var_hat/m_hat^3)*tGrid;

%% Figure 4

figure;

plot(tGrid,H_emp,...
    'LineWidth',2);

hold on;

plot(tGrid,H_theory,...
    'r--',...
    'LineWidth',2);

legend('Simulation','Theory');

xlabel('t');

ylabel('H(t)');

title('Renewal Function');

%% Figure 5

figure;

plot(tGrid,...
    Var_emp,...
    'LineWidth',2);

hold on;

plot(tGrid,...
    Var_theory,...
    'r--',...
    'LineWidth',2);

legend('Simulation','Theory');

xlabel('t');

ylabel('Var(N(t))');

title('Renewal Variance');

%% Renewal CLT

t0=tGrid(end);

N0=Nall(:,end);

Z=(N0-t0/m_hat)/sqrt(t0);

sigmaCLT=sqrt(var_hat/m_hat^3);

figure;

histogram(Z,30,...
    'Normalization','pdf');

hold on;

x=linspace(min(Z),max(Z),300);

y=1/(sqrt(2*pi)*sigmaCLT)...
    *exp(-x.^2/(2*sigmaCLT^2));

plot(x,y,...
    'r',...
    'LineWidth',2);

xlabel('z');

ylabel('Density');

title('Renewal CLT');

%% Errors

errH=max(abs(H_emp-H_theory));

errV=max(abs(Var_emp-Var_theory));

fprintf('\n');
fprintf('============================\n');
fprintf('Renewal Verification\n');
fprintf('============================\n');

fprintf('max|H_emp-H_theory|     = %.6f\n',errH);

fprintf('max|Var_emp-Var_theory| = %.6f\n',errV);

fprintf('\nSimulation Finished.\n');

%% Function

function ISI=simulateISI(...
    gL,gsyn,mu,sigma,...
    Vr,Vth,...
    tau_ref,tau_syn)

dt=0.01;

Tmax=20000;

time=0:dt:Tmax;

V=Vr;

I=mu;

last=-1e10;

spk=[];

for n=1:length(time)

    t=time(n);

    if (t-last)<tau_ref

        V=Vr;

    else

        dW=sqrt(dt)*randn;

        I=I...
            +dt/tau_syn*(-I+mu)...
            +sigma/tau_syn*dW;

        V=V+dt*(-gL*V+gsyn*I);

        if V>=Vth

            spk=[spk,t];

            last=t;

            V=Vr;

        end

    end

end

ISI=diff(spk);

end

