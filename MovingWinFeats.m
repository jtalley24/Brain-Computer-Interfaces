function features = MovingWinFeats(x, fs, winLen, winDisp, featFn)

num_windows = floor((length(x)/fs - winLen)/winDisp + 1);
x_start = 1;
winLen = fs*winLen;
winDisp = fs*winDisp;

for i = 1:num_windows
    x_end = winLen + (i-1)*winDisp;
    feature = x(x_start:x_end);
    features(i) = featFn(feature);
    x_start = x_end - (winLen - winDisp) + 1;
end

end

