// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/image.hpp"

#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#endif

#include "ai_msgs/msg/capture_targets.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "include/ai_msg_manage.h"
#include "include/hand_lmk_output_parser.h"

#ifndef MONO2D_BODY_DET_NODE_H_
#define MONO2D_BODY_DET_NODE_H_

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DnnNodePara;

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::ModelTaskType;
using hobot::dnn_node::ModelRoiInferTask;
using hobot::dnn_node::NV12PyramidInput;

using ai_msgs::msg::PerceptionTargets;

struct HandLmkOutput : public DnnNodeOutput {
  std::shared_ptr<std_msgs::msg::Header> image_msg_header = nullptr;
  // 符合resizer模型限制条件的roi
  std::shared_ptr<std::vector<hbDNNRoi>> valid_rois;
  // 原始roi的索引对应于valid_rois的索引
  std::map<size_t, size_t> valid_roi_idx;

  // 算法推理使用的图像数据，用于本地渲染使用
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;

  ai_msgs::msg::PerceptionTargets::UniquePtr ai_msg;
  ai_msgs::msg::Perf perf_preprocess;
};

struct FeedbackImgInfo {
  std::string image_ = "config/960x544.nv12";
  int img_w = 960;
  int img_h = 544;
  std::vector<std::vector<int32_t>> rois = {{181, 12, 382, 185}, 
                                            {625, 212, 806, 443}};
};

class HandLmkDetNode : public DnnNode {
 public:
  HandLmkDetNode(const std::string &node_name,
                 const NodeOptions &options = NodeOptions());
  ~HandLmkDetNode() override;

 protected:
  int SetNodePara() override;

  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  // 用于预测的图片来源，0：订阅到的image msg；1：本地nv12格式图片
  int feed_type_ = 0;
  FeedbackImgInfo fb_img_info_;

  std::string model_file_name_ = "config/handLMKs.hbm";
  std::string model_name_ = "handLMKs";
  ModelTaskType model_task_type_ = ModelTaskType::ModelRoiInferType;

  int model_input_width_ = -1;
  int model_input_height_ = -1;
  int32_t model_output_count_ = 1;
  const int32_t kps_output_index_ = 0;
  float expand_scale_ = 1.25;
  // resizer model input size limit
  // roi, width & hight must be in range [16, 256)
  int32_t roi_size_max_ = 255;
  int32_t roi_size_min_ = 16;

  int is_sync_mode_ = 0;

  // 使用shared mem通信方式订阅图片
  int is_shared_mem_sub_ = 1;

  int dump_render_img_ = 0;
  int render_count_ = 0;

  std::string ai_msg_pub_topic_name = "/hobot_hand_lmk_detection";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;

  int Feedback();

  int Predict(std::vector<std::shared_ptr<DNNInput>> &inputs,
              const std::shared_ptr<std::vector<hbDNNRoi>> rois,
              std::shared_ptr<DnnNodeOutput> dnn_output);

#ifdef SHARED_MEM_ENABLED
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      sharedmem_img_subscription_ = nullptr;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
#endif

  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 目前只支持订阅原图，可以使用压缩图"/image_raw/compressed" topic
  // 和sensor_msgs::msg::CompressedImage格式扩展订阅压缩图
  std::string ros_img_topic_name_ = "/image_raw";
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  int Render(const std::shared_ptr<NV12PyramidInput> &pyramid,
             std::string result_image,
             std::shared_ptr<std::vector<hbDNNRoi>> &valid_rois,
             std::shared_ptr<LandmarksResult> &lmk_result);

  std::shared_ptr<AiMsgManage> ai_msg_manage_ = nullptr;
  std::string ai_msg_sub_topic_name_ = "/hobot_mono2d_body_detection";
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr
      ai_msg_subscription_ = nullptr;
  void AiMsgProcess(const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg);
  
  int NormalizeRoi(const hbDNNRoi *src, hbDNNRoi *dst,
                  float norm_ratio, uint32_t total_w, uint32_t total_h);

  // 将订阅到的图片数据转成pym之后缓存
  // 在线程中执行推理，避免阻塞订阅IO通道，导致AI msg消息丢失
  std::mutex mtx_img_;
  std::condition_variable cv_img_;
  using CacheImgType = std::pair<std::shared_ptr<HandLmkOutput>,
                                 std::shared_ptr<NV12PyramidInput>>;
  std::queue<CacheImgType> cache_img_;
  size_t cache_len_limit_ = 8;
  void RunPredict();
  std::shared_ptr<std::thread> predict_task_ = nullptr;
};

#endif  // MONO2D_BODY_DET_NODE_H_
