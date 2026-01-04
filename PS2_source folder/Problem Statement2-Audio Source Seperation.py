import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

#audio normalization function
def normalize_audio(audio_data):
    """
    Maximizes the volume of the audio so it is clearly audible.
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data


#loading audio file
audio_path = "mixture.wav"
print("Loading audio...")
y, sr = librosa.load(audio_path, sr=None, mono=True)


#converting audio to frequency domain(STFT)
S_full, phase = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))
print("Analyzing signal patterns...")


#pattern analysis using NN filter
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr, hop_length=512)))

#limiting filter output
S_filter = np.minimum(S_full, S_filter)

#creating soft masks
margin_i = 1.5  
margin_v = 2.0  
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)


#extracting voice and flute
S_foreground = mask_v * S_full 
S_background = mask_i * S_full  
print("Reconstructing audio...")

#reconstructing audio signals
y_voice = librosa.istft(S_foreground * phase)
y_flute = librosa.istft(S_background * phase)
print("Boosting volume...")

#boosting volume and saving files
y_voice_boosted = normalize_audio(y_voice)
y_flute_boosted = normalize_audio(y_flute)

sf.write("voice_loud.wav", y_voice_boosted, sr)
sf.write("flute_loud.wav", y_flute_boosted, sr)
print("Done! Saved 'voice_loud.wav' and 'flute_loud.wav'.")


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)

#visualizing spectograms
librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Original Mixture')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Extracted Voice')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_background, ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Extracted Flute')

plt.tight_layout()
plt.show()