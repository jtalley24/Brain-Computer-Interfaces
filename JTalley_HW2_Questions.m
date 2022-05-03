%%
% <latex>
% \title{BE 521: Homework 2 Questions \\{\normalsize Modeling Neurons} \\{\normalsize Spring 2020}}
% \author{46 points}
% \date{Due: Tuesday, 2/4/2020 11:59 PM}
% \maketitle
% \textbf{Objective:} Computational modeling of neurons. \\
% We gratefully acknowledge Dr. Vijay Balasubramanian (UPenn) for many of
% the questions in this homework.\\
% </latex>

%% 
% <latex>
% \begin{center}
% \author{John Talley \\
%   \normalsize Collaborators: Joseph Iwasyk, John Bellwoar \\}
% \end{center}
% </latex>

%%
% <latex>
% \section{Basic Membrane and Equilibrium Potentials (6 pts)}
% Before undertaking this section, you may find it useful to read pg.
% 153-161 of Dayan \& Abbott's \textit{Theoretical Neuroscience} (the 
% relevant section of which, Chapter 5, is posted with the homework). 
% </latex>

%%
% <latex>
% \begin{enumerate}
%  \item Recall that the potential difference $V_T$ when a mole of ions crosses a cell membrane is defined by the universal gas constant $R = 8.31\; {\rm J/mol\cdot K}$, the temperature $T$ (in Kelvin), and Faraday's constant $F = 96,480 {\rm\ C/mol}$ \[ V_T = \frac{RT}{F} \] Calculate $V_T$ at human physiologic temperature ($37\; ^{\circ} \rm C$). (1 pt)
% </latex>

%%

R = 8.31; % J/mol-K
T = 273.15 + 37; % Kelvin
F = 96480; % C/mol

Vt = R*T/F; % 0.0267 Volts

%%
% <latex>
% \rule{\textwidth}{1pt}
% \textit{Example Latex math commands that uses the align tag to make your equations
% neater. You can also input math into sentences with \$ symbol like $\pi + 1$.}
% \begin{align*}
% E = MC^2 \tag{not aligned}\\
% E = & MC^2 \tag{aligned at = by the \&}\\
% 1 = &\; \frac{2}{2}\tag{aligned at = by \&}
% \end{align*}
% \rule{\textwidth}{1pt}
% </latex>

%%
% <latex>
%  \item Use this value $V_T$ to calculate the Nernst equilibrium potentials 
%  (in mV) for the $\rm K^+$, $\rm Na^+$, and $\rm Cl^-$ ions, given the following 
%  cytoplasm and extracellular concentrations in the squid giant axon: 
%  $\rm K^+$ : (120, 4.5), $\rm Na^+$ : (15, 145), and $\rm Cl^-$ : (12, 120), 
%  where the first number is the cytoplasmic and the second the extracellular 
%  concentration (in mM). (2 pts)
% </latex>

%%

% K+
z = 1; % +1 charge
k_in = 120; 
k_out = 4.5;
E_k = (Vt/z)*log(k_out/k_in);
E_k = E_k*1000; % -87.71 mV

% Na+
z = 1; % +1 charge
na_in = 15; 
na_out = 145;
E_na = (Vt/z)*log(na_out/na_in);
E_na = E_na*1000; % 60.61 mV

% Cl-
z = -1; % -1 charge
cl_in = 12; 
cl_out = 120;
E_cl = (Vt/z)*log(cl_out/cl_in);
E_cl = E_cl*1000; % -61.51 mV


%%
% <latex>
%  \item 
%   \begin{enumerate}
% 	\item Use the Goldmann equation,
% 	  \begin{equation}
% 		V_m = V_T\ln\left( \frac{\rm P_{K}\cdot[K^+]_{out} + P_{NA}\cdot[Na^+]_{out} + P_{Cl}\cdot[Cl^-]_{in}}{\rm P_{K}\cdot[K^+]_{in} + P_{NA}\cdot[Na^+]_{in} + P_{Cl}\cdot[Cl^-]_{out}} \right)
% 	  \end{equation}
% 	to calculate the resting membrane potential, $V_m$, assuming that the ratio of membrane permeabilities $\rm P_K:P_{Na}:P_{Cl}$ is $1.0:0.05:0.45$. Use the ion concentrations given above in Question 1.2. (2 pts)
% </latex>

%%

Pk = 1.0;
Pna = 0.05;
Pcl = 0.45;

Vm_rest = Vt*log((Pk*k_out + Pna*na_out + Pcl*cl_in)/(Pk*k_in + Pna*na_in + Pcl*cl_out)); % Volts
Vm_rest_mv = Vm_rest*1000; % -62.01 mV


%%
% <latex>
% 	\item Calculate the membrane potential at the peak action potential, assuming a ratio of $1.0:12:0.45$, again using the ion concentrations given in Question 1.2. (1 pt)
%   \end{enumerate}
% </latex>

%% 

Pna = 12;

Vm_peak = Vt*log((Pk*k_out + Pna*na_out + Pcl*cl_in)/(Pk*k_in + Pna*na_in + Pcl*cl_out)); % Volts
Vm_peak_mv = Vm_peak*1000; % 42.69 mV


%%
% <latex>
% 	\item The amplitudes of the multi-unit signals in HW0 and local field
% 	potentials (LFP) in HW1 had magnitudes on the order of 10 to 100
% 	microvolts. The voltage at the peak of the action potential (determined
% 	using the Goldman equation above) has a magnitude on the order of 10
% 	millivolts. Briefly explain why we see this difference in magnitude.
% 	Hint1: Voltage is the difference in electric potential between two
% 	points. What are the two points for our voltage measurement in the
% 	multi-unit and LFP signals? What are the two points for the voltage
% 	measurement of the action potential? Hint 2: The resistance of the neuronal membrane is typically much higher than the resistance of the extracellular fluid. (2 pts)
% </latex>

%%

% Because the resistance of the neuronal membrane is much higher than that
% of extracellular fluid, we observe a much greater voltage amplitude of
% the peaks of the action potential as described by the Goldmann equation
% than those of the multi-unit and LFP recordings.  For a transmembrane
% recording, we would measure the electric potential difference, or
% "voltage drop" between an intracellular electrode and one located in the extracellular fluid.  Due to
% a high membrane resistance, this voltage difference is greater than that measured
% between two extracellular points on an axon, as is the case with LFP and
% multi-unit recordings.  In the Hodgkin-Huxley model of the giant squid
% axon, an intracelullar electrode and extracellular reference electrode
% measures this potential as a current is injected and confirms the millivolt scale values seen in
% the values from the Goldmann equation.  In multi-unit and LFP recordings,
% the electrodes measure the extracellular voltage change during
% depolarization as Na+ rushes into the cell compared to another
% extracellular reference point, and due to the low resistance of the
% extracelullar fluid comparatively, the measured amplitude of this signal
% is orders of magnitude lower and on the microvolt scale.

%% 
% <latex>
% \end{enumerate}
% \section{Integrate and Fire Model (38 pts)}
% You may find it useful to read pg.\ 162-166 of Dayan and Abbott for this section. The general differential equation for the integrate and fire model is
% \[ \tau_m\frac{dV}{dt} = V_m - V(t) + R_m I_e(t) \]
% where $\tau_m = 10\, \rm ms$ is the membrane time constant, describing how fast the current is leaking through the membrane, $V_m$ in this case is constant and represents the resting membrane potential (which you have already calculated in question 1.3.a), and $V(t)$ is the actual membrane potential as a function of time. $R_m = 10^7\, \Omega$ is the constant total membrane resistance, and $I_e(t)$ is the fluctuating incoming current. Here, we do not explicitly model the action potentials (that's Hodgkin-Huxley) but instead model the neuron's behavior leading up and after the action potential.
% </latex>


%%
% <latex>
% Use a $\Delta t = 10\, \rm \mu s$ ($\Delta t$ is the discrete analog of the continuous $dt$). Remember, one strategy for modeling differential equations like this is to start with an initial condition (here, $V(0)=V_m$), then calculate the function change (here, $\Delta V$, the discrete analog to $dV$) and then add it to the function (here, $V(t)$) to get the next value at $t+\Delta t$. Once/if the membrane potential reaches a certain threshold ($V_{th} = -50\, \rm mV$), you will say that an action potential has occurred and reset the potential back to its resting value.
% \begin{enumerate}
%  \item Model the membrane potential with a constant current injection (i.e., $I_e(t) = I_e = 2 {\rm nA}$). Plot your membrane potential as a function of time to show at least a handful of ``firings.'' (8 pts)
% </latex>

%%
% SOLVING ANALYTICALLY WITH EQN FROM DAYON & ABBOT

tm = 0.010; % seconds
% V(t); % membrane potential as a function of time
% V(0) = Vm; % initial condition for V(t)
Rm = 10^7; % Ohms
% Ie(t); % fluctuating current as a function of time
Ie = 2e-9; % amps
deltaT = 10e-6; % seconds

% Solve differential 

% tm*dV/dt = Vm - V(t) + Rm*Ie

% Solution given on pg. 164 of Dayan & Abbot, eqn 5.9

% V(t) = Vm + Rm*Ie + (V(0) - Vm - Rm*Ie)*e^(-t/tm)
% V(0) = Vm given
Vth = -0.050; % Volts
t_isi = tm*log(Rm*Ie/(Rm*Ie + Vm_rest - Vth)); % 0.0092 sec to each spike
r_isi = 1/t_isi; % firing rate = 108.9531 Hz
t_to_spike = round(t_isi/deltaT); % 918 samples to spike
V = [];

for i = 0:5 % i = number of spikes to display - 1
    for t = 1:t_to_spike
        V(i*t_to_spike + t) = Vm_rest + Rm*Ie - Rm*Ie*exp(-(t*deltaT)/tm);
        if V(i*t_to_spike + t) >= Vth
            V(i*t_to_spike + t) = Vm_rest;
        end
    end
    t = 1;
end

x = linspace(0,length(V)*deltaT,length(V));
figure;
plot(x,V)
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('6 AP Spikes with Constant Injection Current of 2 nA (Analytical)');

%%
% SOLVING DISCREETLY 

v_samples = 5*t_isi; % show 6 spikes
V(1) = Vm_rest;
for k = 2:v_samples
    dV = (Vm_rest - V(k-1) + Rm*Ie)*deltaT/tm;
    V(k) = V(k-1) + dV;
    if V(k) >= Vth
        V(k) = Vm_rest;
    end
end

x = linspace(0,length(V)*deltaT,length(V));
figure;
plot(x,V)
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('6 AP Spikes with Constant Injection Current of 2 nA (Discreet)');

%%
% <latex>
%  \item Produce a plot of firing rate (in Hz) versus injection current, over the range of 1-4 nA. (4 pts)
% </latex>

%%
% Plot firing rate vs. Injection current Ie

Ie = 1e-9:0.01e-9:4e-9;
t_isi = tm.*log(Rm.*Ie./(Rm.*Ie + Vm_rest - Vth)); % 0.0092 sec to each spike
r_isi = 1./t_isi; % firing rate, in Hz

RmIe = Rm.*Ie;
Vth_Vm = Vth-Vm_rest;

for j = 1:length(Ie)
    if RmIe(j) <= Vth_Vm % from Dayon & Abbot 5.11
        r_isi(j) = 0;
    end
end

figure;
plot(Ie, r_isi);
xlabel('Injection Current (amps)');
ylabel('Firing Rate (Hz)');
title('Firing Rate vs. Injection Current (1-4 nA)');

%%
% <latex>
%  \item \texttt{I521\_A0002\_D001} contains a dynamic current injection in nA. Plot the membrane potential of your neuron in response to this variable injection current. Use Matlab's \texttt{subplot} function to place the plot of the membrane potential above the injection current so that they both have the same time axis. (Hint: the sampling frequency of the current injection data is different from the sampling frequency ($\frac{1}{\Delta t}$) that we used above.) (4 pts)
% </latex>

%%
% SOLVING ANALYTICALLY

dataset_ID = 'I521_A0002_D001';
ieeg_id = 'jtalley';
ieeg_pw = 'jta_ieeglogin.bin';

session = IEEGSession(dataset_ID,ieeg_id,ieeg_pw);

Fs_ie  = session.data.sampleRate; % Hz
Fs_og = 1/deltaT;

durationInUSec = session.data(1).rawChannels(1).get_tsdetails.getDuration;

Ie_var = getvalues(session.data,1:durationInUSec,1);
Ie_var = Ie_var.*10^-9;

t_isi = tm.*log(Rm.*Ie_var./(Rm.*Ie_var + Vm_rest - Vth));
r_isi = 1./t_isi;

v_samples = 50000; % 0.500 seconds of membrane potential samples
n = 1:10:500000; % create array used to downsample Injection Current to match membrane potential frequency
Ie_var_downsampled = Ie_var(n);
ti = 1; % placeholder to set time back to 1 and V to Vm_rest when threshold is reached
for t = 1:v_samples
    V(t) = Vm_rest + Rm*Ie_var(n(t)) - Rm*Ie_var(n(t))*exp(-(ti*deltaT)/tm);
        if V(t) >= Vth
            V(t) = Vm_rest;
            v_samples = v_samples - t;
            ti = 0;
        end
    ti = ti + 1;
end


x = linspace(0,0.5,length(V));
figure;
subplot(2,1,1);
plot(x,V);
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('Membrane Potential with Varying Injection Current (Analytical)');

subplot(2,1,2);
plot(x,Ie_var_downsampled);
xlabel('Time (seconds)');
ylabel('Current (amps)');
title('Injection Current vs. Time');

% V(t) = Vm + Rm.*Ie - Rm*Ie*e^(-t/tm);
% tm*dV/dt = Vm - V(t) + Rm*Ie(t)


%%
% SOLVING DISCREETLY 

v_samples = 0.5/deltaT; % show 0.500 seconds
V(1) = Vm_rest;
for k = 2:v_samples
    dV = (Vm_rest - V(k-1) + Rm*Ie_var_downsampled(k-1))*deltaT/tm;
    V(k) = V(k-1) + dV;
    if V(k) >= Vth
        V(k) = Vm_rest;
    end
end

x = deltaT/v_samples:deltaT:0.5;
figure;
subplot(2,1,1);
plot(x,V);
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('Membrane Potential with Varying Injection Current (Discreet)');

subplot(2,1,2);
plot(x,Ie_var_downsampled);
xlabel('Time (seconds)');
ylabel('Current (amps)');
title('Injection Current vs. Time');

%%
% <latex>
%  \item Real neurons have a refractory period after an action potential that prevents them from firing again right away. We can include this behavior in the model by adding a spike-rate adaptation conductance term, $g_{sra}(t)$ (modeled as a potassium conductance), to the model
%  \[ \tau_m\frac{dV}{dt} = V_m - V(t) - r_m g_{sra}(t)(V(t)-V_K)+ R_m I_e(t) \]
%  where \[ \tau_{sra}\frac{dg_{sra}(t)}{dt} = -g_{sra}(t), \]
%  Every time an action potential occurs, we increase $g_{sra}$ by a certain constant amount, $g_{sra} = g_{sra} + \Delta g_{sra}$. Use $r_m \Delta g_{sra} = 0.06$. Use a conductance time constant of $\tau_{sra} = 100\, \rm ms$, a potassium equilibrium potential of $V_K = -70\, \rm mV$, and $g_{sra}(0) = 0$. (Hint: How can you use the $r_m \Delta g_{sra}$ value to update voltage and conductance separately in your simulation?)
% </latex>

%%
% <latex>
%  \begin{enumerate}
%   \item Implement this addition to the model (using the same other parameters as in question 2.1) and plot the membrane potential over 200 ms. (8 pts)
% </latex>

%%

Vk = -0.070; % Volts
Ie = 2e-9; % Amps, constant for part a
t_sra = 0.100; % seconds
time = 0.200; % seconds
rm_gsra = zeros(1,time/deltaT);
V = zeros(1,time/deltaT);
V(1) = Vm_rest;

for t = 2:time/deltaT
    if V(t-1) >= Vth
        V(t) = Vm_rest;
        rm_gsra(t) = rm_gsra(t-1) + 0.06;
    else
        dV = (Vm_rest - V(t-1) - rm_gsra(t-1)*(V(t-1) - Vk) + Rm*Ie)*deltaT/tm;
        V(t) = V(t-1) + dV;
        d_rm_gsra = -rm_gsra(t-1)*deltaT/t_sra;
        rm_gsra(t) = rm_gsra(t-1) + d_rm_gsra;
    end
end

figure;
x = deltaT/20000:deltaT:0.2;
plot(x,V);
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('Membrane Potential with Spike Rate Adaption Conductance');

%%
% <latex>
%   \item Plot the inter-spike interval (the time between the spikes) of all the spikes that occur in 500 ms. (2 pts)
% </latex>

%%

time = 0.500; % 500 ms

for t = 2:time/deltaT
    if V(t-1) >= Vth
        V(t) = Vm_rest;
        rm_gsra(t) = rm_gsra(t-1) + 0.06;
    else
        dV = (Vm_rest - V(t-1) - rm_gsra(t-1)*(V(t-1) - Vk) + Rm*Ie)*deltaT/tm;
        V(t) = V(t-1) + dV;
        d_rm_gsra = -rm_gsra(t-1)*deltaT/t_sra;
        rm_gsra(t) = rm_gsra(t-1) + d_rm_gsra;
    end
end

figure;
x = linspace(0,0.5,time/deltaT);
plot(x,V);
xlabel('Time (seconds)');
ylabel('Amplitude (volts)');
title('Membrane Potential with Spike Rate Adaption Conductance');

[~,locs] = findpeaks(V);
figure;
isi = deltaT*diff(locs);
events = 1:29;
scatter(events,isi);
xlabel('Spikes');
ylabel('Inter-spike Interval (seconds)');
title('Inter-spike Intervals Over 500 ms');

%%
% <latex>
%   \item Explain how the spike-rate adaptation term we introduced above might be contributing to the behavior you observe in 2.4.b. (2 pts)
% </latex>

%%

% As observed in the plots of spikes and inter-spike intervals over a 500
% ms window, the addition of the gsra(t) spike adaptation increases the
% inter-spike interval distance with each successive action potential and 
% plateaus at an isi of around 0.018 seconds.  This makes sense, because at
% the occurence of each action potential, we add a constant of 0.06 to gsra
% (technically to rm*gsra).  This term is subtracted (negative) in the differential
% equation for dV/dt, so it is reasonable that the isi increases as this
% term increases as it reduces the rate of change of the membrane
% potential, and hence increases the time it takes for the spike to reach
% the threshold value.  As gsra increases with each AP, the rate of change
% of this parameter decreases (becomes more negative), which explains the
% plateauing effect of the increase in inter-spike intervals as time progresses.



%%
% <latex>
%  \end{enumerate}
%  \item Pursue an extension of this basic integrate and fire model. A few ideas are: implement the Integrate-and-Fire-or-Burst Model of Smith et al.\ 2000 (included); implement the Hodgkin-Huxley model (see Dayan and Abbot, pg.\ 173); provide some sort of interesting model of a population of neurons; or perhaps model what an electrode sampling at 200 Hz would record from the signal you produce in question 2.3. Feel free to be creative. 
%  We reserve the right to give extra credit to particularly interesting extensions and will in general be more generous with points for more difficult extensions (like the first two ideas), though it is possible to get full credit for any well-done extension.
%   \begin{enumerate}
% 	\item Briefly describe what your extension is and how you will execute it in code. (6 pts)
% </latex>

%%

% I will extend upon the basic integrate and fire model by applying it to
% the Hodgkin-Huxley Model, detailed beginning on pg. 173 of the Dayan and
% Abbot paper.  In doing so, I will plot the dynamics of voltage
% and gating variables n, m, and h against time over a window
% of 50 ms.  I plan to use MATLAB's ode45 function solver to achieve
% this extension, creating functions for the alpha and beta functions for
% the n, m, and h gating variables.  I will also encode differential
% equations for the n and m variable activation probabilities and the h 
% variable deactivation probability.  The parameters for the Hodgkin-Huxley
% model equation will be sourced directly from the Dayan and Abbot paper.
% Using ode45, I will output a solution array containing the values for
% membrane potential V, and gating variables n, m, and h over a 50 ms
% window, and will plot these values over the same time window.

%%
% <latex>
% 	\item Provide an interesting figure along with an explanation illustrating the extension. (4 pts)
% </latex>

%%

% Parameters
global I cm A gbarL gbarK gbarNa EL EK ENa Vm_rest_mv
I = 0.1; % injected current, Ie/A, mA/mm^2
cm = 0.01; % membrance capacitance uF/mm^2
A = 0.1; % mm^2
gbarL = 0.003; % Leakage conductance mS/mm^2
gbarK = 0.36; % K conductance mS/mm^2
gbarNa = 1.20; % Na conductance mS/mm^2
EL = -54.387; % Leakage reversal potential mV
EK = -77; % K reversal potential mV
ENa = 50; % Na reversal potential mV
Vm_rest_mv = -62.0123; % Resting membrane voltage mV

% Equations (from Dr. Balasubramanian's Theoretical Neuroscience class & original Hodgkin-Huxley Paper 1952)
% 
% im = gbarL*(V(t) - EL) + gbarK*(n^4)*(V(t) - EK) + gbarNa*(h*m^3)*(V(t)-ENa);
% 
% cm*dV/deltaT = -im + Ie/A;

% Activation Probabilities
% dn/deltaT = alpha_n(V(t))*(1 -  n(t)) - beta_n(V(t))*n(t);
% dm/deltaT = alpha_m(V(t))*(1 -  m(t)) - beta_m(V(t))*m(t);
% % Deactivation Probability
% dh/deltaT = alpha_h(V(t))*(1 -  h(t)) - beta_h(V(t))*h(t);

% Steady-State / Initial n, m, h values
n0 = alpha_n(Vm_rest_mv)/(alpha_n(Vm_rest_mv) + beta_n(Vm_rest_mv));
m0 = alpha_m(Vm_rest_mv)/(alpha_m(Vm_rest_mv) + beta_m(Vm_rest_mv));
h0 = alpha_h(Vm_rest_mv)/(alpha_h(Vm_rest_mv) + beta_h(Vm_rest_mv));

y0=[Vm_rest_mv; n0; m0; h0];
tspan = 0:0.001:50; % 50 ms

[time,output] = ode45(@HodgkinHuxley,tspan,y0);

figure;
subplot(2,1,1);
plot(time, output(:,1));
xlabel('Time (ms)')
ylabel('Membrane Potential (mV)');
title('Hodgkin-Huxley Model of an Axon');

subplot(2,1,2);
plot(time, output(:,2));
hold on;
plot(time, output(:,3));
plot(time, output(:,4));
legend('n', 'm', 'h');
xlabel('Time (ms)');
title('Gating Variables');


function a_n  = alpha_n(v)
    a_n = 0.01*(v + 55)/(1 - exp(-0.1*(v + 55)));
end

function a_m = alpha_m(v)
    a_m = 0.1*(v + 40)/(1 - exp(-0.1*(v + 40)));
end

function a_h = alpha_h(v)
    a_h = 0.07*exp(-0.05*(v + 65));
end

function b_n = beta_n(v)
    b_n = 0.125*exp(-0.0125*(v + 65));
end

function b_m = beta_m(v)
    b_m = 4*exp(-0.0556*(v + 65));
end

function b_h = beta_h(v)
    b_h = 1/(1 + exp(-0.1*(v + 35)));
end


function dydt = HodgkinHuxley(t,y) 
    global I cm gbarL gbarK gbarNa EL EK ENa Vm_rest_mv
    % y(1) = V(t), y(2) = n(t), y(3) = m(t), y(4) = h(t)
    dydt = zeros(4,1);
    dydt(1) = (I-(gbarL*(y(1) - EL) + gbarK*(y(2)^4)*(y(1) - EK) + gbarNa*(y(4)*(y(3)^3))*(y(1)-ENa)))/cm; % dV/dT
    dydt(2) = alpha_n(y(1))*(1 -  y(2)) - beta_n(y(1))*y(2); % dn/dt
    dydt(3) = alpha_m(y(1))*(1 -  y(3)) - beta_m(y(1))*y(3); % dm/dt
    dydt(4) = alpha_h(y(1))*(1 -  y(4)) - beta_h(y(1))*y(4); % dh/dt

end

%%

% This extension of the integrate and fire model shows 4 action potential
% spikes in the Hodgkin-Huxley model of the giant squid axon, as well the
% oscillations in the gating variables n, m, and h over a 50 ms timespan.  
% The shapes of these curves and temporal evolution of V, n, m, and h match 
% closely with those displayed on pg. 174 of the Dayan & Abbot paper,
% confirming the successful implementation of this extension of the
% integrate and fire model.

%%
% <latex>
%   \end{enumerate}
% \end{enumerate}
% </latex>


