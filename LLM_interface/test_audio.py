'''This utill is to check whether the sound card works well'''
import queue
import threading
import time
import wave
import Hobot.GPIO as GPIO
import pyaudio

# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# GPIO引脚设置
GPIO.setmode(GPIO.BOARD)  # BOARD物理引脚编码模式
GPIO.setup(11, GPIO.IN)  # 设置引脚11为输入模式


class VoiceAssistant:
    def __init__(self):
        # 音频设备初始化
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
        )

        # 状态管理
        self.is_recording = False
        self.audio_buffer = queue.Queue()
        self.frames = []

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调，处理从音频设备接收到的信息"""
        if self.is_recording:
            self.audio_buffer.put(in_data)
            self.frames.append(in_data)
        return in_data, pyaudio.paContinue

    def start_recording(self):
        """开始录音"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.input_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback
            )
            print("状态：录音中... 🎤")

    def stop_recording(self):
        """停止录音并保存为文件"""
        if self.is_recording:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.audio_buffer.put(None)

            # 保存录音到文件
            with wave.open("test.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.frames))
            print("录音已保存为 test.wav 📁")


    def shutdown(self):
        """清理资源"""
        self.p.terminate()


if __name__ == "__main__":
    assistant = VoiceAssistant()

    try:
        while True:
            if GPIO.input(11) == 1 and not assistant.is_recording:
                assistant.start_recording()
            elif GPIO.input(11) == 0 and assistant.is_recording:
                assistant.stop_recording()
        try:
            assistant.play_audio("test.wav")
        except:
            print("fail to play the audio")
    except KeyboardInterrupt:
        print("程序终止")
        assistant.shutdown()
