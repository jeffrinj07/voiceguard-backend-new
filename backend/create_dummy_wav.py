
import wave
import struct
import math

# Create a 1-second silent WAV file
def create_dummy_wav(filename="test_audio.wav"):
    sample_rate = 22050
    duration = 1.0 # seconds
    num_samples = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1) # Mono
        wav_file.setsampwidth(2) # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        
        # Write silence
        for _ in range(num_samples):
            data = struct.pack('<h', 0)
            wav_file.writeframes(data)
            
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_wav("backend/test_audio.wav")
