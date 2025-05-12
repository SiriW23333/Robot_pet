import pyaudio

def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']}")
    audio.terminate()

if __name__ == "__main__":
    list_audio_devices()