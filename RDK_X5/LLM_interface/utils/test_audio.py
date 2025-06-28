'''This util is to check whether the sound card works well'''
import queue
import threading
import time
import wave
import pyaudio
import numpy as np

# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


class VoiceAssistant:
    def __init__(self):
        # 音频设备初始化
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # 查找并打印所有音频设备信息
        print("可用音频设备列表：")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            print(f"索引: {i}, 名称: {info['name']}, 输入通道数: {info['maxInputChannels']}, 输出通道数: {info['maxOutputChannels']}")

        # 指定设备索引
        self.input_device_index = 2 # 录音设备
        self.output_device_index = 3  # 播放设备

        print(f"使用录音设备索引: {self.input_device_index}")
        print(f"使用播放设备索引: {self.output_device_index}")

        # 状态管理
        self.is_recording = False
        self.frames = []

    def start_recording(self, duration=3):
        """开始录音指定时长"""
        print(f"开始录音 {duration} 秒... 🎤")
        self.is_recording = True
        self.frames = []
        
        # 创建录音流
        self.input_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.input_device_index
        )
        
        # 录音指定时长
        for i in range(0, int(RATE / CHUNK * duration)):
            data = self.input_stream.read(CHUNK)
            self.frames.append(data)
            
            # 显示进度
            progress = (i + 1) / (RATE / CHUNK * duration) * 100
            print(f"\r录音进度: {progress:.1f}%", end="", flush=True)
        
        print("\n录音完成! ✅")
        return self.stop_recording()  # 添加 return，返回文件名

    def stop_recording(self):
        """停止录音并保存为文件"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        
        self.is_recording = False
        
        # 保存录音到文件
        filename = "test_recording.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))
        print(f"录音已保存为 {filename} 📁")
        return filename

    def play_audio(self, filename, volume_gain=10.0):
        """播放音频文件，可调节音量"""
        try:
            print(f"开始播放 {filename}... 🔊 (音量增益: {volume_gain}x)")
            
            # 打开音频文件
            with wave.open(filename, "rb") as wf:
                # 创建播放流
                self.output_stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.output_device_index
                )
                
                # 播放音频
                data = wf.readframes(CHUNK)
                while data:
                    # 将音频数据转换为numpy数组进行音量调整
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    # 放大音量，但要防止溢出
                    amplified_data = np.clip(audio_data * volume_gain, -32768, 32767).astype(np.int16)
                    # 转换回字节数据
                    amplified_bytes = amplified_data.tobytes()
                    
                    self.output_stream.write(amplified_bytes)
                    data = wf.readframes(CHUNK)
                
                # 清理播放流
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
                
            print("播放完成! ✅")
            
        except Exception as e:
            print(f"播放音频失败: {e}")

    def shutdown(self):
        """清理资源"""
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()
        self.p.terminate()


def main():
    assistant = VoiceAssistant()
    
    try:
        # 录音10秒
        filename = assistant.start_recording(duration=10)
        
        # 等待一下
        time.sleep(1)
        
        # 播放录下来的音频
        assistant.play_audio(filename)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        assistant.shutdown()
        print("程序结束")


if __name__ == "__main__":
    main()
