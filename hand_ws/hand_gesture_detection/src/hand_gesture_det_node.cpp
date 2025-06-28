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

#include "include/hand_gesture_det_node.h"

#include <math.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_prefix.hpp"

builtin_interfaces::msg::Time ConvertToRosTime(
    const struct timespec& time_spec) {
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

int CalTimeMsDuration(const builtin_interfaces::msg::Time& start,
                      const builtin_interfaces::msg::Time& end) {
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 -
         start.nanosec / 1000 / 1000;
}

namespace inference {
HandGestureDetNode::HandGestureDetNode(const std::string& node_name,
                                       const NodeOptions& options)
    : DnnNode(node_name, options) {
  this->declare_parameter<int>("is_sync_mode", is_sync_mode_);
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name);
  this->declare_parameter<std::string>("ai_msg_sub_topic_name",
                                       ai_msg_sub_topic_name_);

  this->get_parameter<int>("is_sync_mode", is_sync_mode_);
  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name",
                                   ai_msg_pub_topic_name);
  this->get_parameter<std::string>("ai_msg_sub_topic_name",
                                   ai_msg_sub_topic_name_);

  model_name_ = this->declare_parameter<std::string>("model_name", model_name_);
  is_dynamic_gesture_ = this->declare_parameter<bool>("is_dynamic_gesture",
                                                      is_dynamic_gesture_);
  time_interval_sec_ = this->declare_parameter<float>("time_interval_sec",
                                                     time_interval_sec_);
  threshold_ = this->declare_parameter<float>("threshold", threshold_);
  task_num_ = this->declare_parameter<int>("task_num", task_num_);

  // 获取pkg路径
  std::string pkg_path = ament_index_cpp::get_package_prefix(pkg_name_);
  std::string config_path = pkg_path + "/lib/" + pkg_name_ + "/";
  RCLCPP_WARN(this->get_logger(), "pkg_name: %s, pkg_path: %s, config_path: %s",
    pkg_name_.c_str(), pkg_path.c_str(), config_path.c_str());

  if (model_file_name_.empty() || model_name_.empty()) {
    // file or model name is not set, using default parameters 
    if (is_dynamic_gesture_) {
      model_file_name_ = config_path + default_dynamic_model_file_name_;
      model_name_ = default_dynamic_model_name_;
    } else {
      model_file_name_ = config_path + default_static_model_file_name_;
      model_name_ = default_static_model_name_;
    }
  }

  if (is_dynamic_gesture_) {
    sp_vote_ = std::make_shared<tros::Vote>(tros::VoTeType::TIMEINTERVAL, 30, time_interval_sec_);
  }

  std::stringstream ss;
  ss << "Parameter:"
     << "\n is_sync_mode: " << is_sync_mode_
     << "\n task_num: " << task_num_
     << "\n is_dynamic_gesture: " << is_dynamic_gesture_
     << "\n time_interval_sec: " << time_interval_sec_
     <<"\n threshold: " << threshold_
     << "\n model_file_name: " << model_file_name_
     << "\n model_name: " << model_name_
     << "\n ai_msg_sub_topic_name: " << ai_msg_sub_topic_name_
     << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name;
  RCLCPP_WARN(
      this->get_logger(), "%s", ss.str().c_str());

  if (Init() != 0) {
    RCLCPP_ERROR(this->get_logger(), "Init failed!");
  }

  if (GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(this->get_logger(),
                 "Get model input size fail!");
  } else {
    RCLCPP_INFO(this->get_logger(),
                "The model input width is %d and height is %d",
                model_input_width_,
                model_input_height_);
  }

  GetModelIOInfo();

  gesture_preprocess_ =
      std::make_shared<GesturePreProcess>(gesture_preprocess_config_, is_dynamic_gesture_);

  gesture_postprocess_ = std::make_shared<GesturePostProcess>("", threshold_);

  thread_pool_ = std::make_shared<ThreadPool>();
  thread_pool_->msg_handle_.CreatThread(task_num_);

  RCLCPP_WARN(this->get_logger(),
              "Create subscription with topic_name: %s",
              ai_msg_sub_topic_name_.c_str());
  ai_msg_subscription_ =
      this->create_subscription<ai_msgs::msg::PerceptionTargets>(
          ai_msg_sub_topic_name_,
          10,
          std::bind(
              &HandGestureDetNode::AiMsgProcess, this, std::placeholders::_1));

  RCLCPP_WARN(this->get_logger(),
              "ai_msg_pub_topic_name: %s",
              ai_msg_pub_topic_name.data());
  msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      ai_msg_pub_topic_name, 10);
}

HandGestureDetNode::~HandGestureDetNode() {}

int HandGestureDetNode::SetNodePara() {
  RCLCPP_INFO(this->get_logger(), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type = model_task_type_;
  dnn_node_para_ptr_->task_num = task_num_;
  return 0;
}

int HandGestureDetNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput>& node_output) {
  if (!rclcpp::ok()) {
    return 0;
  }

  if (!node_output ||
      output_index_ >= static_cast<int32_t>(node_output->output_tensors.size())) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid node output");
    return -1;
  }

  auto hand_gesture_output =
      std::dynamic_pointer_cast<HandGestureOutput>(node_output);
  if (!hand_gesture_output) {
    return -1;
  }

  if (!gesture_postprocess_) {
    RCLCPP_ERROR(this->get_logger(),
              "Invalid gesture postprocess");
    return -1;
  }
  std::shared_ptr<GestureRes> gesture_res = nullptr;
  gesture_res = gesture_postprocess_->Execute(hand_gesture_output->output_tensors[0], 
                                              hand_gesture_output->track_id, 
                                              hand_gesture_output->timestamp);

  if (!msg_publisher_) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid msg_publisher_");
    return -1;
  }

  const auto& res = gesture_res;
  if (!is_dynamic_gesture_) {
    // static gesture
    if (res->value_ < static_cast<int>(gesture_type::Background) ||
        res->value_ > static_cast<int>(gesture_type::Awesome)) {
      hand_gesture_output->gesture_res->gesture_res_.push_back(
          gesture_type::Background);
    } else {
      hand_gesture_output->gesture_res->gesture_res_.push_back(
          static_cast<gesture_type>(res->value_));
    }
  } else {
    // dynamic gesture
    if (res->value_ != static_cast<int>(gesture_type::PinchMove) &&
        res->value_ != static_cast<int>(gesture_type::PinchRotateClockwise) &&
        res->value_ != static_cast<int>(gesture_type::PinchRotateAntiClockwise)) {
      hand_gesture_output->gesture_res->gesture_res_.push_back(
          gesture_type::Background);
    } else {
      auto gesture = static_cast<gesture_type>(res->value_);
      // mirror clock and anticlockwise
      if (gesture == gesture_type::PinchRotateClockwise) {
        gesture = gesture_type::PinchRotateAntiClockwise;
      } else if (gesture == gesture_type::PinchRotateAntiClockwise) {
        gesture = gesture_type::PinchRotateClockwise;
      }

      hand_gesture_output->gesture_res->gesture_res_.push_back(gesture);
    }
  }

  hand_gesture_output->gesture_res->prom_.set_value(0);
  hand_gesture_output->gesture_res->is_promised_ = true;

  return 0;
}

void HandGestureDetNode::Publish(
    ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg,
    ai_msgs::msg::Perf perf_preprocess,
    const std::unordered_map<uint64_t, std::shared_ptr<HandGestureRes>>&
        gesture_outputs) {
  // append gesture to ai msg and publish ai msg
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_ai_msg(
      new ai_msgs::msg::PerceptionTargets());
  pub_ai_msg->set__header(msg->header);
  pub_ai_msg->set__fps(msg->fps);
  pub_ai_msg->set__perfs(msg->perfs);
  pub_ai_msg->set__disappeared_targets(msg->disappeared_targets);

  std::stringstream ss;
  ss << "publish msg"
     << ", frame_id: " << pub_ai_msg->header.frame_id
     << ", stamp: " << pub_ai_msg->header.stamp.sec << "_"
     << msg->header.stamp.nanosec << "\n";
  for (const auto& in_target : msg->targets) {
    ai_msgs::msg::Target target;
    target.set__type(in_target.type);
    target.set__rois(in_target.rois);
    target.set__captures(in_target.captures);
    target.set__track_id(in_target.track_id);
    target.set__points(in_target.points);
    target.set__attributes(in_target.attributes);

    bool target_has_hand = false;
    for (const auto& roi : in_target.rois) {
      if (roi.type == "hand") {
        target_has_hand = true;
        break;
      }
    }
    if (target_has_hand &&
        gesture_outputs.find(in_target.track_id) != gesture_outputs.end()) {
      const auto& gesture_res = gesture_outputs.at(in_target.track_id);
      if (gesture_res && !gesture_res->gesture_res_.empty()) {
        for (const gesture_type& res : gesture_res->gesture_res_) {
          ai_msgs::msg::Attribute attr;
          attr.set__type("gesture");
          attr.set__value(static_cast<int>(res));
          target.attributes.emplace_back(attr);
          ss << "\t target id: " << in_target.track_id
             << ", attr type: " << attr.type.data() << ", val: " << attr.value
             << "\n";
        }
      }
    }
    pub_ai_msg->targets.emplace_back(target);
  }

  {
    std::unique_lock<std::mutex> lk(frame_stat_mtx_);
    if (!output_tp_) {
      output_tp_ =
          std::make_shared<std::chrono::high_resolution_clock::time_point>();
      *output_tp_ = std::chrono::system_clock::now();
    }
    auto tp_now = std::chrono::system_clock::now();
    output_frameCount_++;
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - *output_tp_)
                        .count();
    if (interval >= 5000) {
      float out_fps = static_cast<float>(output_frameCount_) /
                      (static_cast<float>(interval) / 1000.0);
      RCLCPP_WARN(this->get_logger(),
                  "Pub smart fps %.2f",
                  out_fps);

      smart_fps_ = round(out_fps);
      output_frameCount_ = 0;
      *output_tp_ = std::chrono::system_clock::now();
    }
  }

  if (smart_fps_ > 0) {
    pub_ai_msg->set__fps(smart_fps_);
  }

  RCLCPP_INFO(
      this->get_logger(), "%s", ss.str().c_str());

  pub_ai_msg->perfs.push_back(perf_preprocess);

  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);

  ai_msgs::msg::Perf perf_predict;
  perf_predict.set__type(model_name_ + "_predict");
  perf_predict.stamp_start = perf_preprocess.stamp_end;
  perf_predict.set__stamp_end(ConvertToRosTime(time_now));
  perf_predict.set__time_ms_duration(
      CalTimeMsDuration(perf_predict.stamp_start, perf_predict.stamp_end));
  pub_ai_msg->perfs.emplace_back(perf_predict);

  ai_msgs::msg::Perf perf_postprocess;
  perf_postprocess.set__type(model_name_ + "_postprocess");
  perf_postprocess.set__stamp_start(ConvertToRosTime(time_now));
  clock_gettime(CLOCK_REALTIME, &time_now);
  perf_postprocess.set__stamp_end(ConvertToRosTime(time_now));
  perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
      perf_postprocess.stamp_start, perf_postprocess.stamp_end));
  pub_ai_msg->perfs.emplace_back(perf_postprocess);

  // 从发布图像到发布AI结果的延迟
  ai_msgs::msg::Perf perf_pipeline;
  perf_pipeline.set__type(model_name_ + "_pipeline");
  perf_pipeline.set__stamp_start(pub_ai_msg->header.stamp);
  perf_pipeline.set__stamp_end(perf_postprocess.stamp_end);
  perf_pipeline.set__time_ms_duration(
      CalTimeMsDuration(perf_pipeline.stamp_start, perf_pipeline.stamp_end));
  pub_ai_msg->perfs.push_back(perf_pipeline);

  msg_publisher_->publish(std::move(pub_ai_msg));
}

int HandGestureDetNode::TenserProcess(
    struct timespec preprocess_time_start,
    ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg,
    std::vector<std::shared_ptr<DNNTensor>> input_tensors,
    std::shared_ptr<std::vector<uint64_t>> track_ids,
    uint64_t timestamp) {
  ai_msgs::msg::Perf perf_preprocess;
  perf_preprocess.set__stamp_start(ConvertToRosTime(preprocess_time_start));
  perf_preprocess.set__type(model_name_ + "_preprocess");

  auto model_manage = GetModel();
  if (!model_manage) {
    RCLCPP_ERROR(this->get_logger(), "Invalid model");
    return -1;
  }

  std::unordered_map<uint64_t, std::shared_ptr<HandGestureRes>> gesture_outputs;
  for (size_t idx = 0; idx < input_tensors.size(); idx++) {
    uint64_t track_id = track_ids->at(idx);

    auto gesture_output = std::make_shared<HandGestureRes>();
    gesture_outputs[track_id] = gesture_output;

    auto dnn_output = std::make_shared<HandGestureOutput>();
    dnn_output->gesture_res = gesture_output;
    dnn_output->gesture_res->is_promised_ = false;
    dnn_output->timestamp = timestamp;
    dnn_output->track_id = track_id;

    hbSysFlushMem(&input_tensors.at(idx)->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    std::vector<std::shared_ptr<DNNTensor>> inputs{input_tensors.at(idx)};

    struct timespec time_now = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_now);
    perf_preprocess.set__stamp_end(ConvertToRosTime(time_now));
    perf_preprocess.set__time_ms_duration(CalTimeMsDuration(
        perf_preprocess.stamp_start, perf_preprocess.stamp_end));

    uint32_t ret = 0;
    // 3. 开始预测
    RCLCPP_DEBUG(this->get_logger(),
                "task_num: %d",
                dnn_node_para_ptr_->task_num);
    ret = Run(
      inputs, dnn_output, is_sync_mode_ == 1 ? true : false);

    // 4. 处理预测结果，如渲染到图片或者发布预测结果
    if (ret != 0) {
      RCLCPP_ERROR(this->get_logger(),
                   "Run predict failed! ret: %d", ret);
      return ret;
    }
  }

  // 等待所有input_tensor都推理完成
  // 一帧中可能包含多个target，即多个input_tensor，每个input_tensor对应一个PostProcess，PostProcess中设置推理完成标志
  // 所有input_tensor都推理完成后才会将此帧数据pub出去
  for (auto& res : gesture_outputs) {
    std::shared_ptr<HandGestureRes>& gesture_info = res.second;
    if (!gesture_info) {
      continue;
    }
    if (!gesture_info->is_promised_) {
      auto fut_ = gesture_info->prom_.get_future();
      int time_out_ms = 50;
      if (fut_.wait_for(std::chrono::milliseconds(time_out_ms)) ==
          std::future_status::ready) {
        gesture_info->is_promised_ = true;
      } else {
        gesture_info->is_promised_ = false;
        continue;
      }
    }
  }

  if (is_dynamic_gesture_ && sp_vote_) {
    // clear cache
    std::vector<uint32_t> disappeared_id_list;
    for (const auto &disappeared_target : msg->disappeared_targets)
    {
      for (const auto &roi : disappeared_target.rois)
      {
          if ("hand" == roi.type)
          {
            disappeared_id_list.push_back(disappeared_target.track_id);
          }
      }
    }
    sp_vote_->ClearCache(disappeared_id_list);

    // vote
    for (auto& gesture_output : gesture_outputs) {
      auto track_id = gesture_output.first;
      auto& gesture_res = gesture_output.second;
      if (gesture_res && !gesture_res->gesture_res_.empty()) {
        for (gesture_type& res : gesture_res->gesture_res_) {
          int in_val = static_cast<int>(res);
          int out_val;
          if (sp_vote_ && sp_vote_->DoProcess(in_val, track_id, out_val) == 0) {
            if (out_val >= static_cast<int>(gesture_type::PinchMove) &&
              out_val <= static_cast<int>(gesture_type::PinchRotateClockwise)) {
              res = static_cast<gesture_type>(out_val);
            }
          }
        }
      }
    }
  }


  if (msg_publisher_) {
    Publish(msg, perf_preprocess, gesture_outputs);
  }

  return 0;
}

void HandGestureDetNode::AiMsgProcess(
    const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg) {
  if (!msg || !rclcpp::ok()) {
    return;
  }

  struct timespec preprocess_time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &preprocess_time_start);

  {
    std::unique_lock<std::mutex> lk(frame_stat_mtx_);
    static auto tp_tp = std::chrono::system_clock::now();
    static int output_frameCount = 0;
    auto tp_now = std::chrono::system_clock::now();
    output_frameCount++;
    auto interval =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_tp)
            .count();
    if (interval >= 5000) {
      float fps = static_cast<float>(output_frameCount) /
                  (static_cast<float>(interval) / 1000.0);
      RCLCPP_WARN(
          this->get_logger(), "Sub smart fps %.2f", fps);
      tp_tp = std::chrono::system_clock::now();
      output_frameCount = 0;
    }
  }

  std::stringstream ss;
  ss << "Recved ai msg"
     << ", frame_id: " << msg->header.frame_id
     << ", stamp: " << msg->header.stamp.sec << "_"
     << msg->header.stamp.nanosec;
  RCLCPP_INFO(
      this->get_logger(), "%s", ss.str().c_str());

  auto pub_msg = [this, msg]() {
    ai_msgs::msg::PerceptionTargets::UniquePtr ai_msg(
        new ai_msgs::msg::PerceptionTargets());
    ai_msg->set__header(msg->header);
    ai_msg->set__fps(msg->fps);
    ai_msg->set__targets(msg->targets);
    ai_msg->set__disappeared_targets(msg->disappeared_targets);
    ai_msg->set__perfs(msg->perfs);
    {
      std::unique_lock<std::mutex> lk(frame_stat_mtx_);
      if (!output_tp_) {
        output_tp_ =
            std::make_shared<std::chrono::high_resolution_clock::time_point>();
        *output_tp_ = std::chrono::system_clock::now();
      }
      auto tp_now = std::chrono::system_clock::now();
      output_frameCount_++;
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - *output_tp_)
                          .count();
      if (interval >= 5000) {
        float out_fps = static_cast<float>(output_frameCount_) /
                        (static_cast<float>(interval) / 1000.0);
        RCLCPP_WARN(this->get_logger(),
                    "Pub smart fps %.2f",
                    out_fps);

        smart_fps_ = round(out_fps);
        output_frameCount_ = 0;
        *output_tp_ = std::chrono::system_clock::now();
      }
    }
    if (smart_fps_ > 0) {
      ai_msg->set__fps(smart_fps_);
    }
    msg_publisher_->publish(std::move(ai_msg));
  };

  std::vector<std::shared_ptr<DNNTensor>> input_tensors;
  std::shared_ptr<std::vector<uint64_t>> track_ids =
      std::make_shared<std::vector<uint64_t>>();
  uint64_t timestamp;
  if (!gesture_preprocess_ ||
      gesture_preprocess_->Execute(
          msg, input_model_info_, input_tensors, *track_ids, timestamp) != 0 ||
      input_tensors.empty()) {
    pub_msg();
    return;
  }

  if (is_sync_mode_) {
    if (TenserProcess(
            preprocess_time_start, msg, input_tensors, track_ids, timestamp) <
        0) {
      pub_msg();
      return;
    }
  } else {
    std::lock_guard<std::mutex> lock(thread_pool_->msg_mutex_);
    if (thread_pool_->msg_handle_.GetTaskNum() >=
        thread_pool_->msg_limit_count_) {
      RCLCPP_WARN(this->get_logger(),
                  "Task Size: %d exceeds limit: %d",
                  thread_pool_->msg_handle_.GetTaskNum(),
                  thread_pool_->msg_limit_count_);
      pub_msg();
      return;
    }

    auto infer_task = [this,
                       preprocess_time_start,
                       msg,
                       input_tensors,
                       track_ids,
                       timestamp,
                       pub_msg]() {
      if (TenserProcess(
              preprocess_time_start, msg, input_tensors, track_ids, timestamp) <
          0) {
        pub_msg();
        return;
      }
    };
    thread_pool_->msg_handle_.PostTask(infer_task);
  }
}

int HandGestureDetNode::GetModelIOInfo() {
  auto model_manage = GetModel();
  if (!model_manage) {
    RCLCPP_ERROR(this->get_logger(), "Invalid model");
    return -1;
  }

  hbDNNHandle_t dnn_model_handle = model_manage->GetDNNHandle();
  int input_num = model_manage->GetInputCount();
  input_model_info_.resize(input_num);
  for (int input_idx = 0; input_idx < input_num; input_idx++) {
    hbDNNGetInputTensorProperties(
        &input_model_info_[input_idx], dnn_model_handle, input_idx);

    std::stringstream ss;
    ss << "input_idx: " << input_idx
       << ", tensorType = " << input_model_info_[input_idx].tensorType
       << ", tensorLayout = " << input_model_info_[input_idx].tensorLayout;
    RCLCPP_WARN(
        this->get_logger(), "%s", ss.str().c_str());
  }

  return 0;
}

}  // namespace inference
