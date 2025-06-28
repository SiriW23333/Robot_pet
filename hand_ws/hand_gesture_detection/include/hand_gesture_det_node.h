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

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ai_msgs/msg/capture_targets.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "rclcpp/rclcpp.hpp"

#include "include/gesture_preprocess.h"
#include "include/gesture_postprocess.h"
#include "threads/threadpool.h"
#include "vote.hpp"

#ifndef HAND_GESTURE_DET_NODE_H_
#define HAND_GESTURE_DET_NODE_H_

namespace inference {

using rclcpp::NodeOptions;

using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::Model;
using hobot::dnn_node::ModelTaskType;

using ai_msgs::msg::PerceptionTargets;

struct HandGestureRes {
  std::promise<int> prom_;
  std::atomic_bool is_promised_;
  std::vector<gesture_type> gesture_res_;
};

struct HandGestureOutput : public DnnNodeOutput {
  uint64_t track_id;
  uint64_t timestamp;
  std::shared_ptr<HandGestureRes> gesture_res = nullptr;
};

struct ThreadPool {
  hobot::CThreadPool msg_handle_;
  std::mutex msg_mutex_;
  int msg_limit_count_ = 10;
};

class HandGestureDetNode : public DnnNode {
 public:
  HandGestureDetNode(const std::string &node_name,
                     const NodeOptions &options = NodeOptions());
  ~HandGestureDetNode() override;

 protected:
  int SetNodePara() override;

  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  std::string pkg_name_ = "hand_gesture_detection";
  std::string default_static_model_file_name_ = "config/gestureDet_8x21.hbm";
  std::string default_static_model_name_ = "gestureDet_8x21";
  std::string default_dynamic_model_file_name_ = "config/gestureDet_32x21.hbm";
  std::string default_dynamic_model_name_ = "gestureDet_32x21";

  std::string model_file_name_ = "";
  std::string model_name_ = "";
  ModelTaskType model_task_type_ = ModelTaskType::ModelInferType;
  bool is_dynamic_gesture_ = false;

  int model_input_width_ = -1;
  int model_input_height_ = -1;
  int32_t model_output_count_ = 1;
  const int32_t output_index_ = 0;

  float threshold_ = 0.5;

  int is_sync_mode_ = 0;
  int task_num_ = 4;

  std::string ai_msg_pub_topic_name = "/hobot_hand_gesture_detection";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;

  std::string gesture_preprocess_config_{""};
  std::shared_ptr<GesturePreProcess> gesture_preprocess_ = nullptr;

  // 模型后处理
  std::shared_ptr<GesturePostProcess> gesture_postprocess_ = nullptr;

  // 模型结构信息, PreProcess需要
  std::vector<hbDNNTensorProperties> input_model_info_;

  std::shared_ptr<std::chrono::high_resolution_clock::time_point> output_tp_ =
      nullptr;
  int output_frameCount_ = 0;
  int smart_fps_ = -1;
  std::mutex frame_stat_mtx_;

  std::shared_ptr<ThreadPool> thread_pool_ = nullptr;

  // int Predict(std::vector<std::shared_ptr<DNNTensor>> &inputs,
  //             std::shared_ptr<DnnNodeOutput> dnn_output);

  std::string ai_msg_sub_topic_name_ = "/hobot_hand_lmk_detection";
  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr
      ai_msg_subscription_ = nullptr;
  void AiMsgProcess(const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg);
  int TenserProcess(struct timespec preprocess_time_start,
                     ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg,
                     std::vector<std::shared_ptr<DNNTensor>> input_tensors,
                     std::shared_ptr<std::vector<uint64_t>> track_ids,
                     uint64_t timestamp);
  void Publish(
      ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg,
      ai_msgs::msg::Perf perf_preprocess,
      const std::unordered_map<uint64_t, std::shared_ptr<HandGestureRes>>
          &gesture_outputs);

  int GetModelIOInfo();

  // 使用0.25秒内的数据投票
  float time_interval_sec_ = 0.25;
  std::shared_ptr<tros::Vote> sp_vote_ = nullptr;
};
}  // namespace inference
#endif  // HAND_GESTURE_DET_NODE_H_
