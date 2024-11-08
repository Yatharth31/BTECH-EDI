import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from safetensors import safe_open
import librosa
from Speech_to_text.speech_to_text import transcribe_audio
import warnings 
from Audio_Enhancement.audio_enhancement import enhance_audio
from Text_to_Text.text_to_text import text_to_text_translation
from Text_to_ISL.convert2isl import convert_to_isl

# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 


if __name__=="__main__":
    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")  # Use any model architecture

    safetensor_path = "./model/model.safetensors"  # Path to your saved safetensors file
    tensors = {}

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    model.load_state_dict(tensors, strict=False)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    print("Model loaded successfully...")
    
    audio_file_path = "./train_marathimale_00020.wav"  # Replace with your audio file path
    result = transcribe_audio(audio_file_path,model,processor)
    print("Transcription:", result)
    
    print("Translation to desired language...")
    input_language = "en"
    output_language = "mr"
    translated_text = text_to_text_translation(result, input_language, output_language)
    print("Translated Text:", translated_text)
    
    # Start two threads for TTS and Text to ISL
    print("TTS...")
    
    print("Text to ISL...")
    convert_to_isl(result)
    print("ISL video generated successfully...")
    