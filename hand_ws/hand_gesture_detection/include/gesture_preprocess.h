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

#ifndef GESTURE_PREPROCESS_H_
#define GESTURE_PREPROCESS_H_

#include <memory>
#include <string>
#include <vector>

#include "ai_msgs/msg/capture_targets.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node_data.h"
#include "include/gesture_data.h"
#include "include/lmks_process.h"

namespace inference {

using hobot::dnn_node::DNNTensor;
using inference::DataType;
using inference::FloatTensor;
using inference::TensorLayout;

class GesturePreProcess {
 public:
  explicit GesturePreProcess(const std::string& json_str, bool is_dynamic_gesture = false) {
    if (!json_str.empty()) {
      // todo 从配置文件中更新配置参数
    }
    if (is_dynamic_gesture) {
      seq_len_ = seq_len_dynamic_;
    } else {
      seq_len_ = seq_len_static_;
    }
    std::stringstream ss;
    ss << "seq_len: " << seq_len_ << ", kps_len: " << kps_len_
       << ", stride: " << stride_ << ", max_gap: " << max_gap_
       << ", buf_len: " << buf_len_ << ", kps_norm_scale: " << kps_norm_scale_
       << ", input_shift: " << input_shift_ << "";
    RCLCPP_INFO(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());

    lmks_proc_.Init(kps_len_,
                    seq_len_,
                    stride_,
                    max_gap_,
                    kps_norm_scale_,
                    norm_kps_conf_,
                    buf_len_);
  }

  ~GesturePreProcess() {}

  int Init(const std::string& json_str);
  int Execute(const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg,
              const std::vector<hbDNNTensorProperties>& input_model_info_,
              std::vector<std::shared_ptr<DNNTensor>>& input_tensors,
              std::vector<uint64_t>& track_ids,
              uint64_t& ts);

 private:
  uint64_t startup_ts_second_ = 0;

  int seq_len_static_ = 8;
  int seq_len_dynamic_ = 32;
  int seq_len_ = 8;
  int kps_len_ = 21;
  int buf_len_ = 100;
  int input_shift_ = 7;
  float max_gap_ = 200;
  float stride_ = 33.3;
  float kps_norm_scale_ = 4.897640403536304;
  float skip_hand_thr_ratio_ = 0.2;
  bool norm_kps_conf_ = false;
  LmksProcess lmks_proc_;
};

}  // namespace inference

#endif  // GESTURE_PREPROCESS_H_
