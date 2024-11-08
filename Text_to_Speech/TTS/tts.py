import torch
from api import TTS
import coqpit

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(TTS().list_models())


# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Clubs and walls and cities grew to be only memories.", speaker_wav="./train_marathimale00020.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="./train_marathimale00020.wav", language="en", file_path="output.wav")