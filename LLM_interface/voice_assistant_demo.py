import asyncio
import base64
import json
import queue
import threading
import time
import Hobot.GPIO as GPIO
import websockets
import pyaudio

# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# GPIO引脚设置
GPIO.setmode(GPIO.BOARD) #BOARD物理引脚编码模式
GPIO.setup(11, GPIO.IN) #设置引脚11为输入模�?

class VoiceAssistant:
    def __init__(self):
        # 音频设备初始�?        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
        )

        # 状态管�?        self.is_recording = False
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


    async def _start_voice_assistant(self):
        ������初始化WebSocket连接并发送会话配�?������
        try:
            self.ws = await websockets.connect(
                "wss://ai-gateway.vei.volces.com/v1/realtime?model=AG-voice-chat-agent",
                extra_headers={"Authorization":sk-4f4d3136c53b4954a87eea49c3d9d0adkjdvm6sozx3s33go}
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
            print(f"会话创建成功：{resp}")

            while True:
                async for message in self.ws:
                    event = json.loads(message)
                    if event.get("type") == "response.audio.delta":
                        self.output_stream.write(base64.b64decode(event["delta"]))
                    elif event.get("type") == "response.done":
                        print(event)
                        self.response_done = True
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))
                    elif event.get("type") == "conversation.item.input_audio_transcription.completed" or \
                            event.get("type") == "response.audio_transcript.delta" or \
                            event.get("type") == "response.audio_transcript.done" or \
                            event.get("type") == "response.output_item.done" or \
                            event.get("type") == "error":
                        print(event)
                        self.chat_history.append(json.dumps(event, ensure_ascii=False))

        except Exception as e:
            print(f"响应处理错误: {str(e)}")
        except Exception as e:
            print(f"初始化连接失�? {str(e)}")
            self.ws = None

    def _start_event_loop(self):
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调"""
        if self.is_recording:
            self.audio_buffer.put(in_data)
        return in_data, pyaudio.paContinue

    def start_recording(self):
        """开始录�?"""
        if not self.is_recording:
            self.is_recording = True
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
            print("状态：录音�?.. 🎤")

    def stop_recording(self):
        """停止录音并发�?"""
        if self.is_recording:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.audio_buffer.put(None)
            print("状态：音频返回�?.. 📤")

    async def send_audio(self):
        """发送音频并实时播放响应"""
        try:
            if not self.ws:
                raise RuntimeError("WebSocket连接未就�?)

            if not self.response_done:
                await self.ws.send(json.dumps({"type": "response.cancel"}))
                self.chat_history.append(json.dumps("=============cancel===========", ensure_ascii=False))

            # 发送录音数�?            while True:
                data = self.audio_buffer.get()
                if data is None:
                    # 采音结束，commit开始大模型推理
                    await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    event = {
                        "type": "response.create",
                        "response": {
                            # "modalities": ["audio"]
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
            if GPIO.input[11] == 1 and not assistant.is_recording:
                assistant.start_recording()
            elif GPIO.input[11] == 0 and assistant.is_recording:
                assistant.stop_recording()
    except KeyboardInterrupt:
        print("程序终止")
        assistant.shutdown()

