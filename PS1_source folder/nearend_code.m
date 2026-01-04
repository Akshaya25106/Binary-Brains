clc;
clear;
close all;

fs = 16000;                 % Sampling frequency
recObj = audiorecorder(fs,16,1);   % 16-bit, mono

disp('Speak now (near-end voice)...');
recordblocking(recObj, 10);   % Record for 5 seconds

nearEnd = getaudiodata(recObj);

% Make the sound louder (important)
nearEnd = nearEnd / max(abs(nearEnd));

% Save audio
audiowrite('near_end.wav', nearEnd, fs);

disp('Near-end recording saved as near_end.wav');


