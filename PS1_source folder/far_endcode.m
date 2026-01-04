fs = 16000;
recObj = audiorecorder(fs,16,1);

disp('Play the far-end sound now...');
recordblocking(recObj, 10);   % record for 10 seconds

farEnd = getaudiodata(recObj);

% Normalize volume
farEnd = farEnd / max(abs(farEnd));

audiowrite('far_end.wav', farEnd, fs);

disp('Far-end recording saved');
sound(farEnd, fs);