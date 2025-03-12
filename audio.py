import whisper

# Load the Whisper model (use a smaller model like 'tiny' or 'small' if memory is an issue)
model = whisper.load_model("tiny")

# Load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("m.wav")
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio
options = whisper.DecodingOptions(fp16=False)
result = whisper.decode(model, mel, options)

# Print the recognized text
print(result.text)

