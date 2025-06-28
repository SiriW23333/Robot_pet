import time
import cv2
import numpy as np
import onnxruntime
from utils.utils import letterbox_image, preprocess_input
from utils.utils_bbox import non_max_suppression, retinaface_correct_boxes
from utils.anchors import Anchors
from utils.config import cfg_mnet
import torch
import csv
import os
from datetime import datetime

class RetinaFaceONNX:
    def __init__(self, onnx_path, input_shape=(640, 640), confidence=0.5, nms_iou=0.45):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = input_shape
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = True
        
        # 生成anchors（需要与训练时一致）
        self.anchors = Anchors(cfg_mnet, image_size=input_shape).get_anchors()
        
    def preprocess(self, image):
        """图像预处理"""
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        
        # letterbox_image可以给图像增加灰条，实现不失真的resize
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        
        # 归一化
        image = preprocess_input(image)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image, im_height, im_width

    def postprocess(self, outputs, im_height, im_width):
        """后处理"""
        loc, conf, landms = outputs
        loc = loc[0]
        conf = conf[0]
        landms = landms[0]
        
        # 只保留人脸类别的置信度
        scores = conf[:, 1]
        
        # 过滤低置信度的检测
        mask = scores > self.confidence
        if not np.any(mask):
            return []
        
        loc = loc[mask]
        scores = scores[mask] 
        landms = landms[mask]
        
        # 先进行anchor解码 - 这是关键步骤
        # 需要将预测的偏移量转换为实际坐标
        from utils.utils_bbox import decode, decode_landm
        variance = [0.1, 0.2]
        
        # 转换为torch tensor进行解码
        loc_tensor = torch.from_numpy(loc)
        landms_tensor = torch.from_numpy(landms)
        anchors_tensor = self.anchors[mask]
        
        # 解码边界框和关键点
        boxes = decode(loc_tensor, anchors_tensor, variance)
        landms_decoded = decode_landm(landms_tensor, anchors_tensor, variance)
        
        # 转回numpy
        boxes = boxes.cpu().numpy()
        landms_decoded = landms_decoded.cpu().numpy()
        scores = scores.reshape(-1, 1)
        
        # 拼接结果 [x1, y1, x2, y2, score, landmarks...]
        result = np.concatenate([boxes, scores, landms_decoded], axis=1)
        
        # NMS
        from torchvision.ops import nms
        boxes_tensor = torch.from_numpy(result[:, :4])
        scores_tensor = torch.from_numpy(result[:, 4])
        keep = nms(boxes_tensor, scores_tensor, self.nms_iou)
        
        if len(keep) == 0:
            return []
        
        result = result[keep.cpu().numpy()]
        
        # 坐标还原到原图尺寸
        scale = [im_width, im_height, im_width, im_height]
        scale_for_landmarks = [im_width, im_height] * 5
        
        # letterbox还原
        if self.letterbox_image:
            result = retinaface_correct_boxes(result, 
                                            np.array([self.input_shape[0], self.input_shape[1]]), 
                                            np.array([im_height, im_width]))
        
        # 应用比例缩放
        result[:, :4] = result[:, :4] * scale
        result[:, 5:] = result[:, 5:] * scale_for_landmarks
        
        return result

    def detect_image(self, image):
        """检测图片"""
        # 预处理
        img_input, im_height, im_width = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: img_input.astype(np.float32)})
        
        # 后处理
        boxes_conf_landms = self.postprocess(outputs, im_height, im_width)
        
        # 绘制结果
        image = np.array(image)
        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            
            # 绘制人脸框
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            # 绘制关键点
            cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)    # 左眼
            cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)  # 右眼
            cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4) # 鼻子
            cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)  # 左嘴角
            cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)  # 右嘴角
            
        return image

def test_onnx_video_performance(onnx_path, video_path=0, video_save_path="", video_fps=25.0):
    """测试ONNX模型在视频模式下的性能"""
    print(f"加载ONNX模型: {onnx_path}")
    retinaface_onnx = RetinaFaceONNX(onnx_path)
    
    # 打开视频捕获
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"无法打开视频源: {video_path}")
        return
    
    # 设置视频保存
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        print(f"视频将保存到: {video_save_path}")
    
    # 性能统计变量
    frame_count = 0
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0
    total_time = 0
    fps_history = []
    
    # 10秒性能统计变量
    ten_sec_frames = 0
    ten_sec_preprocess_time = 0
    ten_sec_inference_time = 0
    ten_sec_postprocess_time = 0
    ten_sec_total_time = 0
    ten_sec_face_count = 0
    
    ref, frame = capture.read()
    if not ref:
        print("无法读取视频帧")
        return
    
    print("开始ONNX模型视频性能测试...")
    print("按 ESC 键退出，按 'i' 键显示详细信息")
    print("将在10秒后自动记录性能数据到CSV文件")
    print("-" * 60)
    
    fps = 0.0
    start_time = time.time()
    ten_sec_start_time = start_time
    
    while True:
        frame_start_time = time.time()
        
        # 读取帧
        ref, frame = capture.read()
        if not ref:
            break
        
        frame_count += 1
        ten_sec_frames += 1
        
        # 格式转换
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 分阶段计时
        # 1. 预处理
        preprocess_start = time.time()
        img_input, im_height, im_width = retinaface_onnx.preprocess(frame_rgb)
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        
        # 2. 推理
        inference_start = time.time()
        outputs = retinaface_onnx.session.run(None, 
                                            {retinaface_onnx.input_name: img_input.astype(np.float32)})
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        # 3. 后处理
        postprocess_start = time.time()
        boxes_conf_landms = retinaface_onnx.postprocess(outputs, im_height, im_width)
        postprocess_end = time.time()
        postprocess_time = postprocess_end - postprocess_start
        
        # 绘制结果
        draw_start = time.time()
        frame_rgb = np.array(frame_rgb)
        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame_rgb, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame_rgb, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # 关键点
            cv2.circle(frame_rgb, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame_rgb, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame_rgb, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame_rgb, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame_rgb, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        draw_end = time.time()
        draw_time = draw_end - draw_start
        
        frame_end_time = time.time()
        frame_total_time = frame_end_time - frame_start_time
        
        # 更新统计
        total_preprocess_time += preprocess_time
        total_inference_time += inference_time
        total_postprocess_time += postprocess_time
        total_time += frame_total_time
        
        # 更新10秒统计
        ten_sec_preprocess_time += preprocess_time
        ten_sec_inference_time += inference_time
        ten_sec_postprocess_time += postprocess_time
        ten_sec_total_time += frame_total_time
        ten_sec_face_count += len(boxes_conf_landms)
        
        # 计算FPS
        fps = (fps + (1.0 / frame_total_time)) / 2
        fps_history.append(1.0 / frame_total_time)
        
        # 检查是否到达10秒
        current_time = time.time()
        if current_time - ten_sec_start_time >= 10.0:
            # 计算10秒内的平均性能
            ten_sec_elapsed = current_time - ten_sec_start_time
            avg_fps = ten_sec_frames / ten_sec_elapsed
            avg_preprocess = ten_sec_preprocess_time / ten_sec_frames * 1000
            avg_inference = ten_sec_inference_time / ten_sec_frames * 1000
            avg_postprocess = ten_sec_postprocess_time / ten_sec_frames * 1000
            avg_total = ten_sec_total_time / ten_sec_frames * 1000
            avg_faces = ten_sec_face_count / ten_sec_frames
            
            # 保存到CSV
            save_performance_to_csv(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_path=onnx_path,
                frames=ten_sec_frames,
                elapsed_time=ten_sec_elapsed,
                avg_fps=avg_fps,
                avg_preprocess=avg_preprocess,
                avg_inference=avg_inference,
                avg_postprocess=avg_postprocess,
                avg_total=avg_total,
                avg_faces=avg_faces
            )
            
            print(f"\n10秒性能数据已保存到CSV文件")
            print(f"平均FPS: {avg_fps:.2f}, 平均推理时间: {avg_inference:.2f}ms")
            
            # 重置10秒统计
            ten_sec_frames = 0
            ten_sec_preprocess_time = 0
            ten_sec_inference_time = 0
            ten_sec_postprocess_time = 0
            ten_sec_total_time = 0
            ten_sec_face_count = 0
            ten_sec_start_time = current_time
        
        # 显示性能信息
        info_text = [
            f"FPS: {fps:.1f}",
            f"Preprocess: {preprocess_time*1000:.1f}ms",
            f"Inference: {inference_time*1000:.1f}ms", 
            f"Postprocess: {postprocess_time*1000:.1f}ms",
            f"Draw: {draw_time*1000:.1f}ms",
            f"Total: {frame_total_time*1000:.1f}ms",
            f"Faces: {len(boxes_conf_landms)}",
            f"10s Countdown: {10.0 - (current_time - ten_sec_start_time):.1f}s"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Frame {frame_count}: FPS={fps:.2f}, "
              f"Preprocess={preprocess_time*1000:.1f}ms, "
              f"Inference={inference_time*1000:.1f}ms, "
              f"Postprocess={postprocess_time*1000:.1f}ms")
        
        cv2.imshow("ONNX Video Detection", frame)
        
        # 保存视频
        if video_save_path != "":
            out.write(frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('i'):  # 显示详细信息
            elapsed_time = current_time - start_time
            print("\n" + "="*60)
            print("详细性能统计:")
            print(f"总帧数: {frame_count}")
            print(f"运行时间: {elapsed_time:.2f}s")
            print(f"平均FPS: {frame_count / elapsed_time:.2f}")
            print(f"平均预处理时间: {total_preprocess_time/frame_count*1000:.2f}ms")
            print(f"平均推理时间: {total_inference_time/frame_count*1000:.2f}ms")
            print(f"平均后处理时间: {total_postprocess_time/frame_count*1000:.2f}ms")
            print(f"平均总时间: {total_time/frame_count*1000:.2f}ms")
            if fps_history:
                print(f"最大FPS: {max(fps_history):.2f}")
                print(f"最小FPS: {min(fps_history):.2f}")
                print(f"FPS标准差: {np.std(fps_history):.2f}")
            print("="*60 + "\n")
    
    # 清理资源
    capture.release()
    if video_save_path != "":
        out.release()
        print(f"视频已保存到: {video_save_path}")
    cv2.destroyAllWindows()
    
    # 最终统计
    total_elapsed = time.time() - start_time
    print("\n最终性能报告:")
    print("-" * 60)
    print(f"总帧数: {frame_count}")
    print(f"总运行时间: {total_elapsed:.2f}s")
    print(f"平均FPS: {frame_count / total_elapsed:.2f}")
    print(f"平均预处理时间: {total_preprocess_time/frame_count*1000:.2f}ms ({total_preprocess_time/total_time*100:.1f}%)")
    print(f"平均推理时间: {total_inference_time/frame_count*1000:.2f}ms ({total_inference_time/total_time*100:.1f}%)")
    print(f"平均后处理时间: {total_postprocess_time/frame_count*1000:.2f}ms ({total_postprocess_time/total_time*100:.1f}%)")
    print(f"平均总处理时间: {total_time/frame_count*1000:.2f}ms")
    if fps_history:
        print(f"最大FPS: {max(fps_history):.2f}")
        print(f"最小FPS: {min(fps_history):.2f}")
        print(f"FPS标准差: {np.std(fps_history):.2f}")

def save_performance_to_csv(timestamp, model_path, frames, elapsed_time, avg_fps, avg_preprocess, avg_inference, avg_postprocess, avg_total, avg_faces):
    """将性能数据保存到CSV文件"""
    csv_file = "onnx_performance_log.csv"
    file_exists = os.path.exists(csv_file)
    
    # CSV文件头
    headers = [
        "Timestamp", "Model_Path", "Frames", "Elapsed_Time(s)", 
        "Avg_FPS", "Avg_Preprocess(ms)", "Avg_Inference(ms)", 
        "Avg_Postprocess(ms)", "Avg_Total(ms)", "Avg_Faces_Per_Frame"
    ]
    
    # 写入CSV文件
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)
            print(f"创建新的CSV文件: {csv_file}")
        
        # 写入数据
        writer.writerow([
            timestamp,
            model_path,
            frames,
            f"{elapsed_time:.2f}",
            f"{avg_fps:.2f}",
            f"{avg_preprocess:.2f}",
            f"{avg_inference:.2f}",
            f"{avg_postprocess:.2f}",
            f"{avg_total:.2f}",
            f"{avg_faces:.2f}"
        ])
    
    print(f"性能数据已添加到: {csv_file}")

if __name__ == "__main__":
    # 配置参数
    onnx_model_path = "/root/Robot_pet/face_ws/retinaface-pytorch/model_output_dir/model_output_dir/retinaface.onnx"
    video_path = 0  # 0表示摄像头，或者指定视频文件路径
    video_save_path = ""  # 保存路径，空字符串表示不保存
    video_fps = 25.0  # 保存视频的FPS
    
    # 检查模型文件
    import os
    if not os.path.exists(onnx_model_path):
        print(f"ONNX模型文件不存在: {onnx_model_path}")
        exit()
    
    # 运行性能测试
    test_onnx_video_performance(onnx_model_path, video_path, video_save_path, video_fps)