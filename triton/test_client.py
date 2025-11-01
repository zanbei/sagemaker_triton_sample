import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import librosa
import sys
import argparse

def load_audio(file_path):
    """Load and preprocess audio file."""
    try:
        # Load audio file with librosa
        audio, sr = librosa.load(file_path, sr=16000)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        sys.exit(1)

def test_inference(audio_path, text_prefix="", repetition_penalty=1.0):
    try:
        # Create client for HTTP protocol
        client = httpclient.InferenceServerClient(url="localhost:8000")

        # Load and preprocess audio
        audio_data = load_audio(audio_path)
        
        # Prepare all required inputs
        inputs = []
        
        # 1. TEXT_PREFIX input
        text_prefix_data = np.array([[text_prefix]], dtype=np.object_)
        text_prefix_input = httpclient.InferInput("TEXT_PREFIX", text_prefix_data.shape, "BYTES")
        text_prefix_input.set_data_from_numpy(text_prefix_data)
        inputs.append(text_prefix_input)
        
        # 2. WAV input
        audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension
        wav_input = httpclient.InferInput("WAV", audio_data.shape, "FP32")
        wav_input.set_data_from_numpy(audio_data)
        inputs.append(wav_input)
        
        # 3. REPETITION_PENALTY input (optional)
        # Adding batch dimension to match expected shape [-1,1]
        rep_penalty_data = np.array([[repetition_penalty]], dtype=np.float32)  # Add batch dimension
        rep_penalty_input = httpclient.InferInput("REPETITION_PENALTY", rep_penalty_data.shape, "FP32")
        rep_penalty_input.set_data_from_numpy(rep_penalty_data)
        inputs.append(rep_penalty_input)

        # Create output tensor
        outputs = [httpclient.InferRequestedOutput("TRANSCRIPTS")]

        # Send inference request
        response = client.infer(model_name="whisper", inputs=inputs, outputs=outputs)

        # Get the transcription
        output_data = response.as_numpy("TRANSCRIPTS")
        print("Transcription:", output_data[0].decode('utf-8'))

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Whisper Triton client")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--text_prefix", type=str, default="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>", 
                        help="Text prefix for transcription")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, 
                        help="Repetition penalty value (default: 1.0)")
    
    args = parser.parse_args()
    
    test_inference(args.audio_path, args.text_prefix, args.repetition_penalty)
