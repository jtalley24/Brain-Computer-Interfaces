%%
% <latex>
% \title{BE 521: Homework 1 \\{\normalsize Exploring Neural Signals} \\{\normalsize Spring 2020}}
% \author{33 points}
% \date{Due: Tuesday 1/28/2020 11:59 PM}
% \maketitle
% \textbf{Objective:} Working with the IEEG Portal to explore different Neural signals
% </latex>

%%
% <latex>
% \section{Seizure Activity (16 pts)} 
% The dataset \texttt{I521\_A0001\_D002} contains an example of human intracranial EEG (iEEG) data displaying seizure activity. It is recorded from a single channel (2 electrode contacts) implanted in the hippocampus of a patient with temporal lobe epilepsy being evaluated for surgery. In these patients, brain tissue where seizures are seen is often resected. You will do multiple comparisons with this iEEG data and the unit activity that you worked with in Homework 0 \texttt{(I521\_A0001\_D001)}. You will have to refer to that homework and/or dataset for these questions.
% \begin{enumerate}
%  \item Retrieve the dataset in MATLAB using the IEEGToolbox and generate a \emph{session} variable as before (No need to report the output this time). What is the sampling rate of this data? What is the maximum frequency of the signal content that we can resolve? (2 pts)
% </latex>

%% 

dataset_ID = 'I521_A0001_D002';
ieeg_id = 'jtalley';
ieeg_pw = 'jta_ieeglogin.bin';

session = IEEGSession(dataset_ID,ieeg_id,ieeg_pw);

Fs  = session.data.sampleRate; % Hz
FsMax = Fs/2; % Hz
channel_labels = session.data.channelLabels(:,1);

% Sampling Rate: 200 Hz
% Max Frequency Resolved: 100 Hz

%%
% <latex>
%  \item How does the duration of this recording compare with the recording from HW0 \texttt{(I521\_A0001\_D001)}? (2 pts)
% </latex>

%%

durationInUSec = session.data(1).rawChannels(1).get_tsdetails.getDuration;
durationInSec = durationInUSec./1e6;

% Duration: 644.995 seconds for HW1 vs 10 seconds from HW0, or 64.4995 times as
% long a recording

%%
% <latex>
%  \item Using the time-series visualization functionality of the IEEG Portal, provide a screenshot of the first 500 ms of data from this recording. (2 pts)
% </latex>

%%

% \includegraphics[scale=0.3]{First500ms.png}\\

%%
% <latex>
%  \item Compare the activity in this sample with the data from HW0.  What differences do you notice in the amplitude and frequency characteristics? (2 pts)
% </latex>

%%

% From the 500 ms screenshots, it is visually clear that the sampling rate
% in HW0 is much greater than that in HW1 (32051 Hz vs. 200 Hz) due to the
% increased noise and granularity of the traces in HW0.  The spikes in
% activity in HW0 also occur over a much shorter transient time period (5 ms vs ~100 ms),
% which justifies the much higher sampling rate to detect these spikes.
% Assuming both samples in microvolts on the y-axis, the amplitude of this
% spike in the HW1 EEG data sample for seizures appears to be slightly
% larger than that in Fried's multiunit EEG recording.

%%
% <latex>
%  \item The unit activity sample in \texttt{(I521\_A0001\_D001)} was high-pass filtered to remove low-frequency content. Assume that the seizure activity in \texttt{(I521\_A0001\_D002)} has not been high-pass filtered. Given that the power of a frequency band scales roughly as $1/f$, how might these differences in preprocessing contribute to the differences you noted in the previous question? (There is no need to get into specific calculations here. We just want general ideas.) (3 pts)
% </latex>

%%

% The activity sample from HW0 was highpass filtered, and as mentioned in
% the previous question, this is clear given the granularity of the trace
% in the 500 ms window.  When the highpass filter is not applied as in the
% HW1 seizure data, low-frequency components are prevalent and the
% resolution of the trace appears lower.  A highpass filter is necessary in
% the trace from HW0 in order to successfully identify the 5 peaks with 5
% ms transients in this window.  If the filter had not been applied, these
% peaks may have been difficult to distinguish from the low-frequency
% components of the signal.  Although the highpass fitler makes these 5
% spikes more visible, the reduction in power due to the filter reduces the
% amplitude of the spikes as well, as power scales with amplitude^2.

%%
session0 = IEEGSession('I521_A0001_D001',ieeg_id,ieeg_pw);
% start time from screenshot
start_time = 4.78; % seconds
end_time = start_time + 0.5; % seconds

data_idx0 = start_time*32051:end_time*32051;
hw0 = getvalues(session0.data,data_idx0,1);

data_idx = 1:0.5*Fs;
hw1 = getvalues(session.data,data_idx,1);

x0 = 0:1/32051:0.5;
x1 = 1/200:1/200:0.5;

% Compare amplitudes
figure;
plot(x0,hw0);
hold on;
plot(x1,hw1);
xlabel('Seconds');
ylabel('Microvolts (uV)');
title('Amplitude Comparison');

%%
% <latex>
%  \item Two common methods of human iEEG are known as electrocorticography (ECoG) and stereoelectroencephalography (SEEG). For either of these paradigms (please indicate which you choose), find and report at least two of the following electrode characteristics: shape, material, size. Please note that exact numbers aren't required, and please cite any sources used. (3 pts)
% </latex>

%%

% A typical ECoG recording device includes an array of evenly spaced
% platinum-iridium electrodes evenly spaced on a biocompatible mesh.  Depending on the
% region of interest in the brain wo which the device will be applied, the
% electrodes may be arranged in a grid or radial pattern, and are evenly spaced.  In the two
% papers explored, the number of electrodes ranged from 32 to 64, and have
% a diameter of 4 mm for each electrode as well as an exposed surface of
% 2.4 mm per electrode, and are usually spaced 1 cm apart. The material of
% the mesh grid is typically a biocompatible silicon.

% Hill, N. J., Gupta, D., Brunner, P., Gunduz, A., Adamo, M. A., Ritaccio, A.,
% & Schalk, G. (2012). Recording human electrocorticographic (ECoG) signals 
% for neuroscientific research and real-time functional cortical mapping. 
% Journal of visualized experiments : JoVE, (64), 3993. doi:10.3791/3993

% Xie, K., Zhang, S., Dong, S. et al. Portable wireless electrocorticography 
% system with a flexible microelectrodes array for epilepsy treatment. Sci 
% Rep 7, 7808 (2017). https://doi.org/10.1038/s41598-017-07823-3


%%
% <latex>
%  \item What is a local field potential? How might the  characteristics of human iEEG electrodes cause them to record local field potentials as opposed to multiunit activity, which was the signal featured in HW0 as recorded from 40 micron Pt-Ir microwire electrodes? (2 pts)
% </latex>

%%

% Local field potential refers to the electrical signal generated in a
% tissue not due to the activity of a single unit or group of units, but
% due to the summed and synchronized electrical activities of the cells in
% the tissue.  Rather than being generated within neuronal cells, this
% summed activity is generated across regions of a tissue when neurons fire
% simultaneously and may interfere with the  electrodes' ability to
% detect multi-unit activities.  The main issues with local field
% potentials (LFP's) itnerfering with multi-unit activity (MUA) detection
% is that the collective firing of neurons in a region of interest alters
% the baseline and the polarity of the signal picked up by an electrode.
% In particular, this effect is aggravated when using larger human iEEG
% electrodes as compared to the 40 micron Pt-Ir microwire electrodes, as
% the human iEEG electrodes are much larger in diameter and thus more likely to pick up
% the LFP's in the region of interest that interfere with the true MUA signals.

%%
% <latex>
% \end{enumerate}
% </latex>

%%
% <latex>
% \section{Evoked Potentials (17 pts)} 
% The data in \texttt{I521\_A0001\_D003} contains an example of a very common type of experiment and neuronal signal, the evoked potential (EP). The data show the response of the whisker barrel cortex region of rat brain to an air puff stimulation of the whiskers. The \texttt{stim} channel shows the stimulation pattern, where the falling edge of the stimulus indicates the start of the air puff, and the rising edge indicates the end. The \texttt{ep} channel shows the corresponding evoked potential. 
% Once again, play around with the data on the IEEG Portal, in particular paying attention to the effects of stimulation on EPs. You should observe the data with window widths of 60 secs as well as 1 sec. Again, be sure to explore the signal gain to get a more accurate picture. Finally, get a sense for how long the trials are (a constant duration) and how long the entire set of stimuli and responses are.
% </latex>

%%
% <latex>
% \begin{enumerate}
%  \item Based on your observations, should we use all of the data or omit some of it? (There's no right answer, here, just make your case either way in a few sentences.) (2 pts)
% </latex>

%%

% Based on viewing the data in 1 sec, 15 sec, and 60 sec windows, we should
% omit some of the data to reduce noise in the evoked potenital (ep)
% channel.  The peaks generally appear to be similar in amplitude and
% transient length (about 1 event per second), but the baseline/background
% noise appears to be fluctuating through the session recording.  To
% accurately identify the spikes in this recording, we should filter the
% data to remove this noise so that a standardized threshold can be used to
% locate events.  Additionally, some of the peaks due to experimental artifacts (such as movement) are greater
% than those of the EP response to stimuli and create a "double peak", so removing these trials
% would prevent false identification of spike events that are not actually
% an evoked potential.

%%
% <latex>
%  \item Retrieve the \texttt{ep} and \texttt{stim} channel data in MATLAB. What is the average latency (in ms) of the peak response to the stimulus onset over all trials? (Assume stimuli occurs at exactly 1 second intervals)(3 pts)
% </latex>

%%

dataset_ID = 'I521_A0001_D003';
ieeg_id = 'jtalley';
ieeg_pw = 'jta_ieeglogin.bin';

session = IEEGSession(dataset_ID,ieeg_id,ieeg_pw)
channel_labels = session.data.channelLabels(:,1);

% In viewing the traces on the IEEG portal in windows of 5 seconds, it is
% clear that the air puff (stimulus) is initated on the second, every
% second throughout all trials.  Thus, we can compare the temporal location
% of each peak to the initiation of the trial's stimulus at each second to
% find the average latency of the peak response to the stimulus over all
% trials.

durationInUSec = session.data(1).rawChannels(1).get_tsdetails.getDuration;
durationInSec = durationInUSec./1e6
% duration is 117.9996 seconds

% start_time = 0; % seconds
end_time = 117.9996; % seconds

Fs  = session.data.sampleRate % Hz
channel_labels = session.data.channelLabels(:,1);

data_idx = 1:end_time*Fs; % Convert from time to indices by
%multiplying by sampling frequency
channel_idx = 1:size(channel_labels,1); % Make a vector of channel indices
ep = getvalues(session.data,data_idx,channel_idx(1)); 
stim = getvalues(session.data,data_idx,channel_idx(2));

% figure;
% x = linspace(start_time,end_time,length(ep));
% plot(x,ep);
% hold on;
% plot(x,stim);
% xlim([start_time end_time]);
% xlabel('Seconds');
% ylabel('uV (microvolts)');
% title('Evoked Potential in Rat Brain in Response to Air Puff Stimulus of Whiskers');

trials = zeros(117,2713);
for i = 1:117
    trial_start = 1 + ((i-1)*Fs);
    trial_end = i*Fs;
    trials(i,:) = ep(trial_start:trial_end);
    [peaks(i), locs(i)] = findpeaks(trials(i,:),'MinPeakDistance',2711); % peaks within 2713 samples/second
end

mean_latency = mean(locs)/Fs;
median_latency = median(locs)/Fs;

% The mean latency comes out to 0.1623 seconds.  However, because there are
% some trials where peaks due to noise exceed the true EP event, it is also
% useful to look at the median latency for a more accurate measure, coming
% out to 0.1349 seconds.


%%
% <latex>
%  \item In neuroscience, we often need to isolate a small neural signal buried under an appreciable amount of noise.  One technique to accomplish this is called the spike triggered average, sometimes called signal averaging. This technique assumes that the neural response to a repetitive stimulus is constant (or nearly so), while the noise fluctuates from trial to trial - therefore averaging the evoked response over many trials will isolate the signal and average out the noise.
%  Construct a spike triggered average plot for the data in \texttt{I521\_A0001\_D003}.  Plot the average EP in red.  Using the commands \texttt{hold on} and \texttt{hold off} as well as \texttt{errorbar} and \texttt{plot}, overlay error bars at each time point on the plot to indicate the standard deviation of the responses at any given time point.  Plot the standard deviation error bars in gray (RGB value: [0.7 0.7 0.7]). Make sure to give a proper legend along with your labels. (4 pts)
% </latex>

%%

STA = zeros(Fs,1);
stds = zeros(Fs,1);
for j = 1:Fs
    STA(j) = mean(trials(:,j));
    stds(j) = std(trials(:,j));
end

x = linspace(0,1,Fs);
figure;
errorbar(x,STA,stds,'Color',[0.7 0.7 0.7]);
hold on;
plot(x,STA,'r');
legend('Error Bars', 'STA Plot');
xlabel('Seconds');
ylabel('uV (microvolts)');
title('Spike Triggered Average, All Trials');

%%
% <latex>
%  \item 
%   \begin{enumerate}
% 	\item We often want to get a sense for the amplitude of the noise in a single trial. Propose a method to do this (there are a few reasonably simple methods, so no need to get too complicated). Note: do not assume that the signal averaged EP is the ``true'' signal and just subtract it from that of each trial, because whatever method you propose should be able to work on the signal from a single trial or from the average of the trials. (4 pts)
% </latex>

%%

% In order to remove and isolate the noise in a single trial, it is
% reasonable to use a lowpass filter to remove the high frequency noise
% components of the signal.  To estimate the amplitude of this, we
% can calculate the difference between the original signal and the
% lowpass-filtered signal in a single 1-second trial, and then calculate
% the mean of this trace to get the average amplitude of noise within the
% trial of interest.

%%
% <latex>
% 	\item Show with a few of the EPs (plots and/or otherwise) that your method gives reasonable results. (1 pt)
% </latex>

%%

% for the 1st trial
x = linspace(0,1,Fs);
signal = trials(1,:);
figure;
plot(x,signal);
signal_lowpass = lowpass(trials(1,:),1,Fs);
hold on;
plot(x,signal_lowpass,'r');
title('Filtered & Unfiltered Signal Traces, Trial 1');
xlabel('Seconds');
ylabel('uV (microvolts)');
legend;

noise = signal - signal_lowpass;
noise = abs(noise);
figure;
plot(x,noise);
disp('Trial 1 Noise Amplitude (uV):');
noise_amp = mean(noise)
SNR = snr(signal,noise);
title('Noise Trace, Trial 1');
xlabel('Seconds');
ylabel('uV (microvolts)');

% for the 30th trial
x = linspace(0,1,Fs);
signal = trials(30,:);
figure;
plot(x,signal);
hold on;
signal_lowpass = lowpass(trials(30,:),1,Fs);
plot(x,signal_lowpass);
title('Filtered & Unfiltered Signal Trace, Trial 30');
xlabel('Seconds');
ylabel('uV (microvolts)');
legend;

noise = signal - signal_lowpass;
noise = abs(noise);
figure;
plot(x,noise);
disp('Trial 30 Noise Amplitude (uV):');
noise_amp = mean(noise)
SNR = snr(signal,noise);
title('Noise Trace, Trial 30');
xlabel('Seconds');
ylabel('uV (microvolts)');

% for the 60th trial
x = linspace(0,1,Fs);
signal = trials(60,:);
figure;
plot(x,signal);
hold on;
signal_lowpass = lowpass(trials(60,:),1,Fs);
plot(x,signal_lowpass);
title('Filtered & Unfiltered Signal Trace, Trial 60');
xlabel('Seconds');
ylabel('uV (microvolts)');
legend;

noise = signal - signal_lowpass;
noise = abs(noise);
figure;
plot(x,noise);
disp('Trial 60 Noise Amplitude (uV):');
noise_amp = mean(noise)
SNR = snr(signal,noise);
title('Noise Trace, Trial 60');
xlabel('Seconds');
ylabel('uV (microvolts)');

%%
% <latex>
% 	\item 
%     \begin{enumerate}
%         \item Apply your method on each individual trial and report the mean noise amplitude across all trials. (1 pt)
% </latex>

%%
for k = 1:117
    signal = trials(k,:);
    signal_lowpass = lowpass(trials(k,:),1,Fs);
    noise = signal - signal_lowpass;
    noise = abs(noise);
    noise_amp(k) = mean(noise);
end

noise_mean = mean(noise_amp);
% Mean Noise Amplitude = 367.3577 uV

%%
% <latex>
%         \item Apply your method on the signal averaged EP and report its noise. (1 pt)
% </latex>

%%

% for the STA
x = linspace(0,1,Fs);
signal = STA;
figure;
plot(x,signal);
hold on;
signal_lowpass = lowpass(STA,1,Fs);
plot(x,signal_lowpass);
title('Filtered & Unfiltered Signal Trace, STA');
xlabel('Seconds');
ylabel('uV (microvolts)');

noise = signal - signal_lowpass;
noise = abs(noise);
figure;
plot(x,noise);
disp('STA Noise Amplitude (uV):');
noise_amp = mean(noise)
SNR = snr(signal,noise);
title('Noise Trace, STA');
xlabel('Seconds');
ylabel('uV (microvolts)');

% STA Noise Amplitude = 36.9045 uV

%%
% <latex>
% 	    \item Do these two values make sense? Explain. (1 pt)
% </latex>

%%

% These values make sense.  The spike triggered average (STA) has a noise
% value about 1/10th that of the average noise across all trials.  Because
% the STA is a plot of the averaged signal across 117 1-second long trials
% at each of 2713 samples within each of these trials, it makes sense that
% this "flattened" signal trace would have noise reduced ~10-fold when
% looking at the mean of all trials instead of treating each noisy trial
% discretely and then simply calculating a mean of all the noise amplitude 
% means as we did in 4.c.i.  In this way, the premise of question 3 is 
% confirmed as the STA successfully isolated a small neural signal buried 
% under noise, and closely matches the filtered signal when plotted
% together.


%%
% <latex>
%     \end{enumerate}
%   \end{enumerate}
% \end{enumerate}
% </latex>

