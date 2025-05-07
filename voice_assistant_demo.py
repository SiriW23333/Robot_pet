import asyncio
import base64
import json
import queue
import threading
import time

import websockets

import gradio as gr
import pyaudio

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
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
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

    async def _start_voice_assistant(self):
        """初始化WebSocket连接并发送会话配置"""
        try:
            self.ws = await websockets.connect(
                "wss://ai-gateway.vei.volces.com/v1/realtime?model=AG-voice-chat-agent",
                extra_headers={"Authorization": "Bearer <VEI AI GATEWAY API KEY>"}
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
            print(f"初始化连接失败: {str(e)}")
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
        """开始录音"""
        if not self.is_recording:
            self.is_recording = True
            self.input_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback
            )
            # 提交asr任务
            asyncio.run_coroutine_threadsafe(
                self.send_audio(),
                self.event_loop
            )
            return "状态：录音中... 🎤"

    def stop_recording(self):
        """停止录音并发送"""
        if self.is_recording:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.audio_buffer.put(None)
            return "状态：音频返回中... 📤"

    async def send_audio(self):
        """发送音频并实时播放响应"""
        try:
            if not self.ws:
                raise RuntimeError("WebSocket连接未就绪")

            if not self.response_done:
                await self.ws.send(json.dumps({"type": "response.cancel"}))
                self.chat_history.append(json.dumps("=============cancel===========", ensure_ascii=False))

            # 发送录音数据
            while True:
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

    async def _handle_messages(self):
        """独立处理服务器响应的协程"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                if event.get("type") == "response.audio.delta":
                    self.output_stream.write(base64.b64decode(event["delta"]))
        except Exception as e:
            print(f"响应处理错误: {str(e)}")

    def shutdown(self):
        """清理资源"""
        self.p.terminate()
        self.event_loop.stop()

    def toggle_recording(self):
        """切换录音状态的回调函数"""
        if assistant.is_recording:
            status_text = self.stop_recording()
            new_button_text = "开始录音 🎤"
        else:
            status_text = self.start_recording()
            new_button_text = "停止录音 ⏹"

        return new_button_text, status_text

    def get_chat_history(self):
        """获取对话记录"""
        return "\n".join(self.chat_history)


# 创建界面
with gr.Blocks(title="语音助手") as demo:
    assistant = VoiceAssistant()

    with gr.Row():
        status = gr.Textbox(label="状态", value="准备就绪 ✅", interactive=False)

    with gr.Row():
        record_btn = gr.Button("开始录音 🎤")

    with gr.Row():
        chat_history = gr.Textbox(
            label="对话记录",
            lines=15,
            max_lines=20,
            interactive=False,
            autoscroll=True,  # 自动滚动到底部
            value=lambda: assistant.get_chat_history(),
            every=0.2  # 每0.2秒（200毫秒）刷新一次
        )

    # 事件绑定
    record_btn.click(
        fn=assistant.toggle_recording,
        outputs=[record_btn, status]
    )

if __name__ == "__main__":
    demo.launch()
