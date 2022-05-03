%%
% <latex>
% \title{BE 521: Homework 0 Questions\\{\normalsize Introduction}\\{\normalsize Spring 2020}}
% \author{ John Talley }
% \date{Due: Thursday 1/23/2020 11:59 PM}
% \maketitle
% \textbf{Objective:} Working with the IEEG Portal, basic matlab commands, publishing LaTeX
% </latex>

%%
% <latex>
% \section{Unit Activity (15 pts)} 
% The dataset \texttt{I521\_A0001\_D001} contains an example of multiunit human iEEG data recorded by Itzhak Fried and colleagues at UCLA using 40 micron platinum-iridium electrodes.
% Whenever you get new and potentially unfamiliar data, you should always play around with it: plot it, zoom in and out, look at the shape of individual items of interest (here, the spikes). The spikes here
% will be events appx. 5 ms in duration with amplitudes significantly greater than surrounding background signal.
% \begin{enumerate}
%  \item Using the time-series visualization functionality of the IEEG
%  Portal find a single time-window containing 5 spikes (use a window width
%  of 500 ms). The signal gain should be adjusted so that the spikes can be seen in entirety. Give a screenshot of the IEEG Portal containing the requested plot.  Remember to reference the LaTeX tutorial if you need help with how to do this in LaTeX. (2 pts)\\
% </latex>

%% 
% Include screenshot:

% \includegraphics[scale=0.3]{C:\Users\Jack\Documents\Penn\BE 521\HW0\output\5spikes.png}\\

%%
% <latex>
%  \item Instantiate a new IEEGSession in MATLAB with the
%  \texttt{I521\_A0001\_D001} dataset into a reference variable called
%  \emph{session} (Hint: refer to the IEEGToolbox manual, class tutorial, or the built-in \emph{methods} commands in the \emph{IEEGSession} object - i.e., \emph{session.methods}). Print the output of \emph{session} here. (1 pt)\\
% </latex>

%% 

% instantiate a new session

dataset_ID = 'I521_A0001_D001';
ieeg_id = 'jtalley';
ieeg_pw = 'jta_ieeglogin.bin';

session = IEEGSession(dataset_ID,ieeg_id,ieeg_pw)

%% 
% <latex>
%  \item What is the sampling rate of the recording? You can find this
%  information by exploring the fields in the \emph{session} data structure
%  you generated above. Give your answer in Hz. (2 pts)\\
% </latex>

%%

% access sampling rate from session data
Fs  = session.data.sampleRate % Hz
channel_labels = session.data.channelLabels(:,1);

%% 
% <latex>
%  \item How long (in seconds) is this recording? (1 pt)\\
% </latex>

%%

% retrieve duration from session data and convert to seconds
durationInUSec = session.data(1).rawChannels(1).get_tsdetails.getDuration;
durationInSec = durationInUSec./1e6

%% 
% <latex>
%  \item 
%  \begin{enumerate}
%     \item Using the \emph{session.data.getvalues} method retrieve the
%     data from the time-window you plotted in Q1.1 and re-plot this data
%     using MATLAB's plotting functionality. Note that the amplitude of the EEG signals from the portal is measured in units of $\mu V$ (microvolts), so label your y-axis accordingly. 
%     (NOTE: Always make sure to include the correct units and labels in your plots. This goes for the rest of this and all subsequent homeworks.). (3 pts)\\
% </latex>

%%

% start time from screenshot
start_time = 4.78; % seconds
end_time = start_time + 0.5; % seconds

data_idx = start_time*Fs:end_time*Fs; % Convert from time to indices by
%multiplying by sampling frequency
channel_idx = 1:size(channel_labels,1); % Make a vector of channel indices
data_matrix = getvalues(session.data,data_idx,channel_idx); 

figure;
x = linspace(start_time,end_time,length(data_matrix));
plot(x,data_matrix);
xlim([start_time end_time]);
xlabel('Seconds');
ylabel('uV (microvolts)');
title('EEG Signal over 500 ms Window');


%% 
% <latex>
% 	\item Write a short bit of code to detect the times of each spike peak
% 	(i.e., the time of the maximum spike amplitude) within your
% 	time-window. Plot an 'x' above each spike peak that you detected superimposed on the plot from Q1.5a. (Hint: find where the slope of the signal changes from positive to negative and the signal is also above threshold.) (4 pts)\\
% </latex>

%%

% using findpeaks method from Signal Processing Toolbox (used in Fuccillo
% Lab during past research)

[~,locs_spikes] = findpeaks(data_matrix,'MinPeakHeight',50,...
                                    'MinPeakDistance',0.005);%peaks above 50 uV and 5 ms
figure;
hold on;
plot(x,data_matrix)
locations_spikes = (locs_spikes./Fs) + start_time;
plot(locations_spikes,data_matrix(locs_spikes),'rx','MarkerFaceColor','r')
xlim([start_time end_time])
xlabel('Seconds');
ylabel('uV (microvolts)');
title('EEG Signal over 500 ms Window');

%% 
% <latex>
% 	\item How many spikes do you detect in the entire data sample? (1 pt)\\
% </latex>

%%
start_t = 0; % seconds
end_t = 10; % seconds

data_idx = (start_t*Fs +1):end_t*Fs; % Convert from time to indices by 
%multiplying by sampling frequency
data_all = getvalues(session.data,data_idx,channel_idx); 

[~,locs_spikes] = findpeaks(data_all,'MinPeakHeight',50,...
                                    'MinPeakDistance',0.005);%peaks above 50 uV and 5 ms
                                
length(locs_spikes) % 32 spikes with a threshold of 50 uV and 5 ms transient

%% 
% <latex>
% \end{enumerate}
% 	\item Content Question- In the assigned reading, you 
%   learned about different methods to obtain and localize neural signals for BCIs.
%   Describe the naming convention for the International 10-20 system for EEG recording. Specifically, what do the
% 	letters refer to and what can you infer from the parity (even vs. odd)
% 	of the number at a given site? (1 pt)\\
% </latex>

%%
% The 10-20 naming system refers to the most frequently used electrode
% positions at 10%, 20%, 20%, 20%, 20%, and 10% of the nasion-inion
% distance (or, from the bridge of the nose to the lower crest of the back
% of the skull).  The electrodes are equally spaced both left to right and
% front to back.  The letters Fp, F, C, P, O, and T represent the pre-frontal,
% frontal, central, parietal, occipital, and temproal locations of the
% electrodes.  Even numbers (2,4,6,8) refer to electrodes placed on the
% right hemisphere, odd numbers (1,3,5,7) refer to the left hemisphere, and
% Z refers to the center-line.

%% 
% <latex>
% \end{enumerate}
% </latex>
