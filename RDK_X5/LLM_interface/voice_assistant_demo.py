import asyncio
import base64
import json
import queue
import threading
import time
import Hobot.GPIO as GPIO
import websockets
import pyaudio
import numpy as np

# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# GPIO引脚设置
GPIO.setmode(GPIO.BOARD) #BOARD物理引脚编码模式
GPIO.setup(11, GPIO.IN) #设置引脚11为输入模式


class VoiceAssistant:
    def __init__(self):
        # 音频设备初始化
        self.p = pyaudio.PyAudio()
        self.input_stream = None

        # 指定输入声卡索引为2，输出声卡索引为3
        self.input_device_index = 2
        self.output_device_index = 3

        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=self.output_device_index
        )

        # 状态管理
        self.is_recording = False
        self.response_done = True
        self.audio_buffer = queue.Queue()
        self.chat_history = []

        # WebSocket连接
        self.ws = None
        self.event_loop = asyncio.new_event_loop()
        
        # 启动事件循环线程
        threading.Thread(
            target=self._start_event_loop,
            daemon=True
        ).start()

        # 初始化WebSocket连接
        asyncio.run_coroutine_threadsafe(
            self._start_voice_assistant(),
            self.event_loop
        )

    def amplify_audio(self, audio_data, amplification_factor=10):
        """音频音量放大函数"""
        try:
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 放大音量
            amplified_array = audio_array.astype(np.float32) * amplification_factor
            
            # 防止溢出，限制在int16范围内
            amplified_array = np.clip(amplified_array, -32768, 32767)
            
            # 转换回int16并返回字节数据
            return amplified_array.astype(np.int16).tobytes()
        except Exception as e:
            print(f"音频放大错误: {str(e)}")
            return audio_data  # 如果出错，返回原始数据

    async def _start_voice_assistant(self):
        """初始化WebSocket连接并发送会话配置"""
        try:
            self.ws = await websockets.connect(
                "wss://ai-gateway.vei.volces.com/v1/realtime?model=AG-voice-chat-agent",
                extra_headers={"Authorization":"sk-4f4d3136c53b4954a87eea49c3d9d0adkjdvm6sozx3s33go"}
            )
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": 0.8,
                    "input_audio_transcription": {
                        "model": "any"
                    },
                }
            }))
            resp = await self.ws.recv()

            while True:
                async for message in self.ws:
                    event = json.loads(message)
                    if event.get("type") == "response.audio.delta":
                        # 解码音频数据
                        audio_data = base64.b64decode(event["delta"])
                        # 放大音量10倍后播放
                        amplified_audio = self.amplify_audio(audio_data, 10)
                        self.output_stream.write(amplified_audio)
                    elif event.get("type") == "response.done":
                        print("AI回应完成")
                        self.response_done = True
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))
                    elif event.get("type") == "conversation.item.input_audio_transcription.completed":
                        # 显示用户说话内容
                        transcript = event.get("transcript", "")
                        if transcript:
                            print(f"用户: {transcript}")
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))
                    elif event.get("type") == "response.audio_transcript.delta":
                        # 显示AI回应文本
                        delta = event.get("delta", "")
                        if delta:
                            print(delta, end="", flush=True)
                    elif event.get("type") == "response.audio_transcript.done":
                        print()  # 换行
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))
                    elif event.get("type") == "response.output_item.done":
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))
                    elif event.get("type") == "error":
                        print(f"错误: {event}")
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))

        except Exception as e:
            print(f"响应处理错误: {str(e)}")
        except Exception as e:
            print(f"初始化连接失败: {str(e)}")
            self.ws = None

    def _start_event_loop(self):
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调，处理从音频设备接收到的信息"""
        if self.is_recording:
            self.audio_buffer.put(in_data)
        return in_data, pyaudio.paContinue

    def start_recording(self):
        """开始录音"""
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer = queue.Queue()  # 每次新建队列，避免残留
            self.input_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=self.input_device_index,
                stream_callback=self.audio_callback
            )
            # 提交asr任务
            asyncio.run_coroutine_threadsafe(
                self.send_audio(),
                self.event_loop
            )
            print("录音中...")

    def stop_recording(self):
        """停止录音并发送"""
        if self.is_recording:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.audio_buffer.put(None)  # 及时通知 send_audio 结束
            print("处理中...")

    async def send_audio(self):
        """发送音频并实时播放响应"""
        try:
            if not self.ws:
                raise RuntimeError("WebSocket连接未就绪")

            if not self.response_done:
                await self.ws.send(json.dumps({"type": "response.cancel"}))

            # 发送录音数据
            while True:
                try:
                    data = self.audio_buffer.get(timeout=1)  # 避免永久阻塞
                except queue.Empty:
                    if not self.is_recording:
                        break
                    continue
                if data is None:
                    await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    event = {
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"]
                        }
                    }
                    await self.ws.send(json.dumps(event))
                    self.response_done = False
                    break

                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode("utf-8")
                }))

        except Exception as e:
            print(f"通信异常: {str(e)}")

    def shutdown(self):
        """清理资源"""
        self.p.terminate()
        self.event_loop.stop()



if __name__ == "__main__":
    assistant = VoiceAssistant()

    try:
        while True:
            if GPIO.input(11) == 1 and not assistant.is_recording:
                assistant.start_recording()
            elif GPIO.input(11) == 0 and assistant.is_recording:
                assistant.stop_recording()
    except KeyboardInterrupt:
        print("程序终止")
        assistant.shutdown()

