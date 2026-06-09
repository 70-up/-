%  E-I Network
%  Piecewise Linear Stability Analysis

clear;
clc;
close all;

%% Parameters

tauE = 10;

MEE = 4;
MEI = 3;

MIE = 4;
MII = 1;

hE = 0.5;
hI = 0;

%% Critical tauI

tauI_star = tauE*(MII+1)/(MEE-1);

disp('Critical tauI =');
disp(tauI_star);

tauI_list = [4 tauI_star 10];

%% Time settings

dt = 0.01;

Tmax = 500;

time = 0:dt:Tmax;

%% Time series

figure

for k = 1:length(tauI_list)

    tauI = tauI_list(k);

    vE = 0.1;
    vI = 0.1;

    VE = zeros(size(time));
    VI = zeros(size(time));

    for n = 1:length(time)

        xE = MEE*vE - MEI*vI + hE;
        xI = MIE*vE - MII*vI + hI;

        RE = max(xE,0);
        RI = max(xI,0);

        dvE = (-vE + RE)/tauE;
        dvI = (-vI + RI)/tauI;

        vE = vE + dt*dvE;
        vI = vI + dt*dvI;

        VE(n) = vE;
        VI(n) = vI;

    end

    subplot(3,1,k)

    plot(time,VE,'LineWidth',1.5)

    hold on

    plot(time,VI,'LineWidth',1.5)

    xlabel('t','Interpreter','none')

    ylabel('Rate','Interpreter','none')

    title(['tauI = ',num2str(tauI)],'Interpreter','none')

    legend('vE','vI','Interpreter','none')

    grid on

end

%% Phase portrait

tauI = 10;

vE = 0.1;
vI = 0.1;

VE = zeros(size(time));
VI = zeros(size(time));

for n = 1:length(time)

    xE = MEE*vE - MEI*vI + hE;
    xI = MIE*vE - MII*vI + hI;

    RE = max(xE,0);
    RI = max(xI,0);

    dvE = (-vE + RE)/tauE;
    dvI = (-vI + RI)/tauI;

    vE = vE + dt*dvE;
    vI = vI + dt*dvI;

    VE(n)=vE;
    VI(n)=vI;

end

idx = round(length(VE)/2):length(VE);

figure

plot(VE(idx),VI(idx),'LineWidth',2)

xlabel('vE','Interpreter','none')

ylabel('vI','Interpreter','none')

title('Phase Portrait','Interpreter','none')

grid on

%% Stability indicator versus tauI

tauI_scan = linspace(2,15,150);

Indicator = zeros(size(tauI_scan));

for k = 1:length(tauI_scan)

    tauI = tauI_scan(k);

    vE = 0.1;
    vI = 0.1;

    VE = zeros(size(time));

    for n = 1:length(time)

        xE = MEE*vE - MEI*vI + hE;
        xI = MIE*vE - MII*vI + hI;

        RE = max(xE,0);
        RI = max(xI,0);

        dvE = (-vE + RE)/tauE;
        dvI = (-vI + RI)/tauI;

        vE = vE + dt*dvE;
        vI = vI + dt*dvI;

        VE(n)=vE;

    end

    tail = VE(round(end/2):end);

    Indicator(k) = std(tail);

end

figure

plot(tauI_scan,Indicator,'LineWidth',2)

hold on

xline(tauI_star,'--r','LineWidth',1.5)

xlabel('tauI','Interpreter','none')

ylabel('std(vE)','Interpreter','none')

title('Stability Indicator versus tauI','Interpreter','none')

grid on

%% Dependence on hE

tauI = 10;

hE_scan = linspace(-2,3,200);

Indicator = zeros(size(hE_scan));

for k = 1:length(hE_scan)

    hE = hE_scan(k);

    vE = 0.1;
    vI = 0.1;

    VE = zeros(size(time));

    for n = 1:length(time)

        xE = MEE*vE - MEI*vI + hE;
        xI = MIE*vE - MII*vI + hI;

        RE = max(xE,0);
        RI = max(xI,0);

        dvE = (-vE + RE)/tauE;
        dvI = (-vI + RI)/tauI;

        vE = vE + dt*dvE;
        vI = vI + dt*dvI;

        VE(n)=vE;

    end

    tail = VE(round(end/2):end);

    Indicator(k)=std(tail);

end

figure

plot(hE_scan,Indicator,'LineWidth',2)

xlabel('hE','Interpreter','none')

ylabel('std(vE)','Interpreter','none')

title('Effect of hE','Interpreter','none')

grid on

