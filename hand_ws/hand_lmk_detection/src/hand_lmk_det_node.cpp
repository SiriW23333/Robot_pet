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

#include "include/hand_lmk_det_node.h"

#include <math.h>
#include <unistd.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ai_msgs/msg/capture_targets.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "include/ai_msg_manage.h"
#include "include/hand_lmk_output_parser.h"
#include "rclcpp/rclcpp.hpp"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

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

HandLmkDetNode::HandLmkDetNode(const std::string& node_name,
                               const NodeOptions& options)
    : DnnNode(node_name, options) {
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<int>("is_sync_mode", is_sync_mode_);
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name);
  this->declare_parameter<int>("dump_render_img", dump_render_img_);
  this->declare_parameter<std::string>("ai_msg_sub_topic_name",
                                       ai_msg_sub_topic_name_);

  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<int>("is_sync_mode", is_sync_mode_);
  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name",
                                   ai_msg_pub_topic_name);
  this->get_parameter<int>("dump_render_img", dump_render_img_);
  this->get_parameter<std::string>("ai_msg_sub_topic_name",
                                   ai_msg_sub_topic_name_);
  std::stringstream ss;
  ss << "Parameter:"
     << "\n feed_type(0:sub, 1:local): " << feed_type_
     << "\n is_sync_mode_: " << is_sync_mode_
     << "\n model_file_name_: " << model_file_name_
     << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name
     << "\n is_shared_mem_sub: " << is_shared_mem_sub_
     << "\n dump_render_img: " << dump_render_img_
     << "\n ai_msg_sub_topic_name_: " << ai_msg_sub_topic_name_;

  RCLCPP_WARN(this->get_logger(), "%s", ss.str().c_str());

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

  if (1 == feed_type_) {
    Feedback();
  } else {
    predict_task_ = std::make_shared<std::thread>(
        std::bind(&HandLmkDetNode::RunPredict, this));

    ai_msg_manage_ = std::make_shared<AiMsgManage>();

    RCLCPP_INFO(this->get_logger(),
                "ai_msg_pub_topic_name: %s",
                ai_msg_pub_topic_name.data());
    msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
        ai_msg_pub_topic_name, 10);

    RCLCPP_INFO(rclcpp::get_logger("hand lmk ai msg sub"),
                "Create subscription with topic_name: %s",
                ai_msg_sub_topic_name_.c_str());
    ai_msg_subscription_ =
        this->create_subscription<ai_msgs::msg::PerceptionTargets>(
            ai_msg_sub_topic_name_,
            10,
            std::bind(
                &HandLmkDetNode::AiMsgProcess, this, std::placeholders::_1));

    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      RCLCPP_WARN(this->get_logger(),
                  "Create hbmem_subscription with topic_name: %s",
                  sharedmem_img_topic_name_.c_str());
      sharedmem_img_subscription_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              sharedmem_img_topic_name_,
              rclcpp::SensorDataQoS(),
              std::bind(&HandLmkDetNode::SharedMemImgProcess,
                        this,
                        std::placeholders::_1));
#else
      RCLCPP_ERROR(this->get_logger(), "Unsupport shared mem");
#endif
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "Create subscription with topic_name: %s",
                  ros_img_topic_name_.c_str());
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_topic_name_,
              10,
              std::bind(
                  &HandLmkDetNode::RosImgProcess, this, std::placeholders::_1));
    }
  }
}

HandLmkDetNode::~HandLmkDetNode() {
  std::unique_lock<std::mutex> lg(mtx_img_);
  cv_img_.notify_all();
  lg.unlock();

  if (predict_task_ && predict_task_->joinable()) {
    predict_task_->join();
    predict_task_.reset();
  }
}

int HandLmkDetNode::SetNodePara() {
  RCLCPP_INFO(this->get_logger(), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type = model_task_type_;
  dnn_node_para_ptr_->task_num = 4;
  return 0;
}

int HandLmkDetNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput>& node_output) {
  if (!rclcpp::ok()) {
    return 0;
  }

  if (!node_output) {
    RCLCPP_ERROR(this->get_logger(), "Invalid node output");
    return -1;
  }

  if (!msg_publisher_ && !feed_type_) {
    RCLCPP_ERROR(this->get_logger(), "Invalid msg_publisher_");
    return -1;
  }

  auto hand_lmk_output = std::dynamic_pointer_cast<HandLmkOutput>(node_output);
  if (!hand_lmk_output) {
    return -1;
  }
  if (node_output->output_tensors.empty()) {
    msg_publisher_->publish(std::move(hand_lmk_output->ai_msg));
    return -1;
  }

  if (node_output->rt_stat->fps_updated) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "input fps: %.2f, out fps: %.2f, "
        "infer time ms: %d, post process time ms: %d",
        node_output->rt_stat->input_fps,
        node_output->rt_stat->output_fps,
        node_output->rt_stat->infer_time_ms,
        node_output->rt_stat->parse_time_ms);
  }


  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);

  // 1. 解析模型输出向量
  auto parser = std::make_shared<HandLmkOutputParser>();
  auto lmk_val = std::make_shared<LandmarksResult>();
  if (hand_lmk_output->valid_rois != nullptr) {
    if (parser->Parse(lmk_val, hand_lmk_output->output_tensors[0], hand_lmk_output->valid_rois) < 0) {
      ai_msgs::msg::PerceptionTargets::UniquePtr& msg = hand_lmk_output->ai_msg;
      msg_publisher_->publish(std::move(msg));
      return -1;
    }
  }
  
  if (!lmk_val) {
    return -1;
  }

  std::stringstream ss;
  ss << "Output from";
  if (hand_lmk_output->image_msg_header && hand_lmk_output->valid_rois) {
    ss << ", frame_id: " << hand_lmk_output->image_msg_header->frame_id
       << ", stamp: " << hand_lmk_output->image_msg_header->stamp.sec << "_"
       << hand_lmk_output->image_msg_header->stamp.nanosec
       << ", hand rois size: " << hand_lmk_output->valid_rois->size()
       << ", hand rois idx size: " << hand_lmk_output->valid_roi_idx.size()
       << ", hand lmk size: " << lmk_val->values.size();
  }
  RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

  // 2. 渲染模型结果
  if (hand_lmk_output->pyramid) {
    if (feed_type_ || (dump_render_img_ && render_count_ % 30 == 0)) {
      render_count_ = 0;
      std::string image_name;
      switch(feed_type_) {
        case 1: image_name = "render.jpg"; break;
        case 0: image_name = "render_" +
            std::to_string(hand_lmk_output->image_msg_header->stamp.sec) + "." +
            std::to_string(hand_lmk_output->image_msg_header->stamp.nanosec) + ".jpg"; break;
      }
      Render(hand_lmk_output->pyramid, image_name, hand_lmk_output->valid_rois, lmk_val);
    }
    render_count_++;
  }

  // 本地模式不需要发布推理话题消息
  if (feed_type_) {
    return 0;
  }
  
  // 3. 发布模型推理话题消息
  ai_msgs::msg::PerceptionTargets::UniquePtr& msg = hand_lmk_output->ai_msg;

  if (lmk_val->values.size() != hand_lmk_output->valid_rois->size() ||
      hand_lmk_output->valid_rois->size() !=
          hand_lmk_output->valid_roi_idx.size()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Check hand lmk outputs fail");
    msg_publisher_->publish(std::move(msg));
    return 0;
  }

  if (msg) {
    ai_msgs::msg::PerceptionTargets::UniquePtr ai_msg(
        new ai_msgs::msg::PerceptionTargets());
    ai_msg->set__header(msg->header);
    ai_msg->set__disappeared_targets(msg->disappeared_targets);

    if (node_output->rt_stat) {
      ai_msg->set__fps(round(node_output->rt_stat->output_fps));
    }

    int hand_roi_idx = 0;
    const std::map<size_t, size_t>& valid_roi_idx =
        hand_lmk_output->valid_roi_idx;
    for (const auto& in_target : msg->targets) {
      // 缓存target的body等kps
      std::vector<ai_msgs::msg::Point> tar_points;
      for (const auto& pts : in_target.points) {
        tar_points.push_back(pts);
      }

      ai_msgs::msg::Target target;
      target.set__type(in_target.type);
      target.set__rois(in_target.rois);
      target.set__attributes(in_target.attributes);
      target.set__captures(in_target.captures);
      target.set__track_id(in_target.track_id);

      for (const auto& roi : in_target.rois) {
        RCLCPP_DEBUG(this->get_logger(),
                     "roi.type: %s",
                     roi.type.c_str());
        if ("hand" == roi.type) {
          if (valid_roi_idx.find(hand_roi_idx) == valid_roi_idx.end()) {
            RCLCPP_INFO(this->get_logger(),
                        "This hand is filtered! hand_roi_idx %d is unmatch "
                        "with roi idx",
                        hand_roi_idx);
            std::stringstream ss;
            ss << "valid_roi_idx: ";
            for (auto idx : valid_roi_idx) {
              ss << idx.first << " " << idx.second << "\n";
            }
            RCLCPP_DEBUG(
                this->get_logger(), "%s", ss.str().c_str());
            continue;
          }

          auto hand_valid_roi_idx = valid_roi_idx.at(hand_roi_idx);
          if (hand_valid_roi_idx >= lmk_val->values.size()) {
            RCLCPP_ERROR(this->get_logger(),
                         "hand lmk outputs %d unmatch with roi idx %d",
                         lmk_val->values.size(),
                         hand_valid_roi_idx);
            break;
          }
          const Landmarks& lmks = lmk_val->values.at(hand_valid_roi_idx);
          ai_msgs::msg::Point hand_lmk;
          hand_lmk.set__type("hand_kps");
          for (const Point& lmk : lmks) {
            geometry_msgs::msg::Point32 pt;
            pt.set__x(lmk.x);
            pt.set__y(lmk.y);
            hand_lmk.point.emplace_back(pt);
          }
          // 缓存target的hand kps
          tar_points.push_back(hand_lmk);

          hand_roi_idx++;
        }
      }
      target.set__points(tar_points);
      ai_msg->targets.emplace_back(target);
    }

    for (const auto& target : ai_msg->targets) {
      std::stringstream ss;
      ss << "target id: " << target.track_id
         << ", rois size: " << target.rois.size()
         << " points size: " << target.points.size() << " ";
      if (!target.rois.empty()) {
        ss << " roi type: " << target.rois.front().type << " ";
      }
      if (!target.points.empty()) {
        ss << " point type: " << target.points.front().type << " ";
      }
      RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    ai_msg->set__perfs(msg->perfs);

    hand_lmk_output->perf_preprocess.set__time_ms_duration(
        CalTimeMsDuration(hand_lmk_output->perf_preprocess.stamp_start,
                          hand_lmk_output->perf_preprocess.stamp_end));
    ai_msg->perfs.push_back(hand_lmk_output->perf_preprocess);

    // predict
    if (hand_lmk_output->rt_stat) {
      ai_msgs::msg::Perf perf;
      perf.set__type(model_name_ + "_predict_infer");
      perf.set__stamp_start(
          ConvertToRosTime(hand_lmk_output->rt_stat->infer_timespec_start));
      perf.set__stamp_end(
          ConvertToRosTime(hand_lmk_output->rt_stat->infer_timespec_end));
      perf.set__time_ms_duration(hand_lmk_output->rt_stat->infer_time_ms);
      ai_msg->perfs.push_back(perf);

      perf.set__type(model_name_ + "_predict_parse");
      perf.set__stamp_start(
          ConvertToRosTime(hand_lmk_output->rt_stat->parse_timespec_start));
      perf.set__stamp_end(
          ConvertToRosTime(hand_lmk_output->rt_stat->parse_timespec_end));
      perf.set__time_ms_duration(hand_lmk_output->rt_stat->parse_time_ms);
      ai_msg->perfs.push_back(perf);
    }

    ai_msgs::msg::Perf perf_postprocess;
    perf_postprocess.set__type(model_name_ + "_postprocess");
    perf_postprocess.set__stamp_start(ConvertToRosTime(time_now));
    clock_gettime(CLOCK_REALTIME, &time_now);
    perf_postprocess.set__stamp_end(ConvertToRosTime(time_now));
    perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
        perf_postprocess.stamp_start, perf_postprocess.stamp_end));
    ai_msg->perfs.emplace_back(perf_postprocess);

    // 从发布图像到发布AI结果的延迟
    ai_msgs::msg::Perf perf_pipeline;
    perf_pipeline.set__type(model_name_ + "_pipeline");
    perf_pipeline.set__stamp_start(ai_msg->header.stamp);
    perf_pipeline.set__stamp_end(perf_postprocess.stamp_end);
    perf_pipeline.set__time_ms_duration(
        CalTimeMsDuration(perf_pipeline.stamp_start, perf_pipeline.stamp_end));
    ai_msg->perfs.push_back(perf_pipeline);

    msg_publisher_->publish(std::move(ai_msg));
  } else {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid ai msg!");
    msg_publisher_->publish(std::move(hand_lmk_output->ai_msg));
    return -1;
  }
  return 0;
}

int HandLmkDetNode::Predict(
    std::vector<std::shared_ptr<DNNInput>>& inputs,
    const std::shared_ptr<std::vector<hbDNNRoi>> rois,
    std::shared_ptr<DnnNodeOutput> dnn_output) {
  RCLCPP_DEBUG(this->get_logger(),
               "task_num: %d",
               dnn_node_para_ptr_->task_num);

  RCLCPP_INFO(this->get_logger(),
              "inputs.size(): %d, rois->size(): %d",
              inputs.size(),
              rois->size());

  return Run(inputs,
             dnn_output,
             rois,
             is_sync_mode_ == 1 ? true : false);
}

void HandLmkDetNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  if (!img_msg || !rclcpp::ok()) {
    return;
  }

  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  std::stringstream ss;
  ss << "Recved img encoding: " << img_msg->encoding
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step
     << ", frame_id: " << img_msg->header.frame_id
     << ", stamp: " << img_msg->header.stamp.sec << "_"
     << img_msg->header.stamp.nanosec
     << ", data size: " << img_msg->data.size();
  RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
  // 1. 将图片处理成模型输入数据类型DNNInput
  // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
  std::shared_ptr<NV12PyramidInput> pyramid = nullptr;
  if ("nv12" == img_msg->encoding) {
    // std::string fname = "img_" + std::to_string(img_msg->header.stamp.sec)
    //  + "." + std::to_string(img_msg->header.stamp.nanosec) + ".nv12";
    // std::ofstream ofs(fname);
    // ofs.write(reinterpret_cast<const char*>(img_msg->data.data()),
    //   img_msg->data.size());

    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
        reinterpret_cast<const char*>(img_msg->data.data()),
        img_msg->height,
        img_msg->width,
        img_msg->height,
        img_msg->width);
  } else {
    RCLCPP_ERROR(this->get_logger(), "Unsupport img encoding: %s",
    img_msg->encoding.data());
  }
  if (!pyramid) {
    RCLCPP_ERROR(this->get_logger(), "Get Nv12 pym fail");
    return;
  }
  
  RCLCPP_WARN(this->get_logger(), "prepare input");
  // 2. 创建推理输出数据
  auto dnn_output = std::make_shared<HandLmkOutput>();
  // 将图片消息的header填充到输出数据中，用于表示推理输出对应的输入信息
  dnn_output->image_msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->image_msg_header->set__stamp(img_msg->header.stamp);
  // 将当前的时间戳填充到输出数据中，用于计算perf
  dnn_output->perf_preprocess.stamp_start.sec = time_start.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_start.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");

  if (dump_render_img_) {
    dnn_output->pyramid = pyramid;
  }

  // 3. 将准备好的输入输出数据存进缓存
  std::unique_lock<std::mutex> lg(mtx_img_);
  if (cache_img_.size() > cache_len_limit_) {
    CacheImgType img_msg = cache_img_.front();
    cache_img_.pop();
    auto drop_dnn_output = img_msg.first;
    std::string ts =
        std::to_string(drop_dnn_output->image_msg_header->stamp.sec) + "." +
        std::to_string(drop_dnn_output->image_msg_header->stamp.nanosec);
    RCLCPP_INFO(this->get_logger(),
                "drop cache_img_ ts %s",
                ts.c_str());
    // 可能只有图像消息，没有对应的AI消息
    if (drop_dnn_output->ai_msg) {
      msg_publisher_->publish(std::move(drop_dnn_output->ai_msg));
    }
  }
  CacheImgType cache_img = std::make_pair<std::shared_ptr<HandLmkOutput>,
                                          std::shared_ptr<NV12PyramidInput>>(
      std::move(dnn_output), std::move(pyramid));
  cache_img_.push(cache_img);
  cv_img_.notify_one();
  lg.unlock();
}

#ifdef SHARED_MEM_ENABLED
void HandLmkDetNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!img_msg || !rclcpp::ok()) {
    return;
  }

  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  // dump recved img msg
  // std::ofstream ofs("img_" + std::to_string(img_msg->index) + "." +
  // std::string(reinterpret_cast<const char*>(img_msg->encoding.data())));
  // ofs.write(reinterpret_cast<const char*>(img_msg->data.data()),
  //   img_msg->data_size);

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step << ", index: " << img_msg->index
     << ", stamp: " << img_msg->time_stamp.sec << "_"
     << img_msg->time_stamp.nanosec << ", data size: " << img_msg->data_size;
  RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

  // 1. 将图片处理成模型输入数据类型DNNInput
  // 使用图片生成pym，NV12PyramidInput为DNNInput的子类
  std::shared_ptr<NV12PyramidInput> pyramid = nullptr;
  if ("nv12" ==
      std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))) {
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
        reinterpret_cast<const char*>(img_msg->data.data()),
        img_msg->height,
        img_msg->width,
        img_msg->height,
        img_msg->width);
  } else {
    RCLCPP_INFO(this->get_logger(),
                "Unsupported img encoding: %s",
                img_msg->encoding);
  }
  if (!pyramid) {
    RCLCPP_ERROR(this->get_logger(), "Get Nv12 pym fail!");
    return;
  }

  // 2. 创建推理输出数据
  auto dnn_output = std::make_shared<HandLmkOutput>();
  // 将图片消息的header填充到输出数据中，用于表示推理输出对应的输入信息
  dnn_output->image_msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->image_msg_header->set__stamp(img_msg->time_stamp);
  // 将当前的时间戳填充到输出数据中，用于计算perf
  dnn_output->perf_preprocess.stamp_start.sec = time_start.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_start.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");

  if (dump_render_img_) {
    dnn_output->pyramid = pyramid;
  }

  // 3. 将准备好的输入输出数据存进缓存
  std::unique_lock<std::mutex> lg(mtx_img_);
  if (cache_img_.size() > cache_len_limit_) {
    CacheImgType img_msg = cache_img_.front();
    cache_img_.pop();
    auto drop_dnn_output = img_msg.first;
    std::string ts =
        std::to_string(drop_dnn_output->image_msg_header->stamp.sec) + "." +
        std::to_string(drop_dnn_output->image_msg_header->stamp.nanosec);
    RCLCPP_INFO(this->get_logger(),
                "drop cache_img_ ts %s",
                ts.c_str());
    // 可能只有图像消息，没有对应的AI消息
    if (drop_dnn_output->ai_msg) {
      msg_publisher_->publish(std::move(drop_dnn_output->ai_msg));
    }
  }
  CacheImgType cache_img = std::make_pair<std::shared_ptr<HandLmkOutput>,
                                          std::shared_ptr<NV12PyramidInput>>(
      std::move(dnn_output), std::move(pyramid));
  cache_img_.push(cache_img);
  cv_img_.notify_one();
  lg.unlock();
}
#endif

int HandLmkDetNode::Render(const std::shared_ptr<NV12PyramidInput>& pyramid,
                           std::string result_image,
                           std::shared_ptr<std::vector<hbDNNRoi>> &valid_rois,
                           std::shared_ptr<LandmarksResult> &lmk_result) {
  static cv::Scalar colors[] = {
      cv::Scalar(255, 0, 0),    // red
      cv::Scalar(255, 255, 0),  // yellow
      cv::Scalar(0, 255, 0),    // green
      cv::Scalar(0, 0, 255),    // blue
  };

  char* y_img = reinterpret_cast<char*>(pyramid->y_vir_addr);
  char* uv_img = reinterpret_cast<char*>(pyramid->uv_vir_addr);
  auto height = pyramid->height;
  auto width = pyramid->width;
  auto img_y_size = height * width;
  auto img_uv_size = img_y_size / 2;
  char* buf = new char[img_y_size + img_uv_size];
  memcpy(buf, y_img, img_y_size);
  memcpy(buf + img_y_size, uv_img, img_uv_size);
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, buf);
  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, CV_YUV2BGR_NV12);
  delete[] buf;
  auto& mat = bgr;

  RCLCPP_WARN(this->get_logger(),
              "h w: %d %d,  mat: %d %d",
              height,
              width,
              mat.cols,
              mat.rows);

  // render kps
  if (lmk_result && valid_rois) {
    // auto landmarks_result = std::dynamic_pointer_cast<LandmarksResult>(
    //     lmk_result->outputs.at(kps_output_index_));

    RCLCPP_WARN(this->get_logger(),
                "landmarks_result->values.size: %d",
                lmk_result->values.size());

    for (auto& rect : *valid_rois) {
      cv::rectangle(mat,
                    cv::Point(rect.left, rect.top),
                    cv::Point(rect.right, rect.bottom),
                    cv::Scalar(255, 0, 0),
                    3);
    }

    size_t lmk_num = lmk_result->values.size();
    for (size_t idx = 0; idx < lmk_num; idx++) {
      const auto& lmk = lmk_result->values.at(idx);
      auto& color = colors[idx % 4];
      for (const auto& point : lmk) {
        cv::circle(mat, cv::Point(point.x, point.y), 3, color, 3);
      }
    }
  }

  RCLCPP_INFO(this->get_logger(),
              "Draw result to file: %s",
              result_image.c_str());
  cv::imwrite(result_image, mat);
  return 0;
}

int HandLmkDetNode::Feedback() {
  if (access(fb_img_info_.image_.c_str(), R_OK) == -1) {
    RCLCPP_ERROR(this->get_logger(),
                 "Image: %s not exist!",
                 fb_img_info_.image_.c_str());
    return -1;
  }

  std::ifstream ifs(fb_img_info_.image_, std::ios::in | std::ios::binary);
  if (!ifs) {
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int len = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  char* data = new char[len];
  ifs.read(data, len);

  std::shared_ptr<NV12PyramidInput> pyramid = nullptr;
  pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
      reinterpret_cast<const char*>(data),
      fb_img_info_.img_h,
      fb_img_info_.img_w,
      fb_img_info_.img_h,
      fb_img_info_.img_w);
  delete[] data;
  if (!pyramid) {
    RCLCPP_ERROR(this->get_logger(),
                 "Get Nv12 pym fail with image: %s",
                 fb_img_info_.image_.c_str());
    return -1;
  }

  auto rois = std::make_shared<std::vector<hbDNNRoi>>();
  for (int i = 0; i < fb_img_info_.rois.size(); i++) {
    hbDNNRoi roi;

    roi.left = fb_img_info_.rois[i][0];
    roi.top = fb_img_info_.rois[i][1];
    roi.right = fb_img_info_.rois[i][2];
    roi.bottom = fb_img_info_.rois[i][3];

    // roi's left and top must be even, right and bottom must be odd
    roi.left += (roi.left % 2 == 0 ? 0 : 1);
    roi.top += (roi.top % 2 == 0 ? 0 : 1);
    roi.right -= (roi.right % 2 == 1 ? 0 : 1);
    roi.bottom -= (roi.bottom % 2 == 1 ? 0 : 1);
    RCLCPP_DEBUG(this->get_logger(),
                "input hand roi: %d %d %d %d",
                roi.left,
                roi.top,
                roi.right,
                roi.bottom);

    rois->push_back(roi);
  }

  // 2. 使用pyramid创建DNNInput对象inputs
  // inputs将会作为模型的输入通过RunInferTask接口传入
  std::vector<std::shared_ptr<DNNInput>> inputs;
  for (size_t i = 0; i < rois->size(); i++) {
    inputs.push_back(pyramid);
  }
  auto dnn_output = std::make_shared<HandLmkOutput>();
  dnn_output->valid_rois = rois;
  dnn_output->valid_roi_idx[0] = 0;
  dnn_output->pyramid = pyramid;

  auto model_manage = GetModel();
  if (!model_manage) {
    RCLCPP_ERROR(this->get_logger(), "Invalid model");
    return -1;
  }

  uint32_t ret = 0;
  // 3. 开始预测
  ret = Predict(inputs, rois, dnn_output);

  // 4. 处理预测结果，如渲染到图片或者发布预测结果
  if (ret != 0) {
    return -1;
  }
    
  return 0;
}

void HandLmkDetNode::AiMsgProcess(
    const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg) {
  if (!msg || !rclcpp::ok() || !ai_msg_manage_) {
    return;
  }

  std::stringstream ss;
  ss << "Recved ai msg"
     << ", frame_id: " << msg->header.frame_id
     << ", stamp: " << msg->header.stamp.sec << "_"
     << msg->header.stamp.nanosec;
  RCLCPP_INFO(
      rclcpp::get_logger("hand lmk ai msg sub"), "%s", ss.str().c_str());

  ai_msg_manage_->Feed(msg);
}

void HandLmkDetNode::RunPredict() {
  while (rclcpp::ok()) {
    std::unique_lock<std::mutex> lg(mtx_img_);
    cv_img_.wait(lg, [this]() { return !cache_img_.empty() || !rclcpp::ok(); });
    if (cache_img_.empty()) {
      continue;
    }
    if (!rclcpp::ok()) {
      break;
    }
    CacheImgType img_msg = cache_img_.front();
    cache_img_.pop();
    lg.unlock();

    auto dnn_output = img_msg.first;
    auto pyramid = img_msg.second;

    std::string ts =
        std::to_string(dnn_output->image_msg_header->stamp.sec) + "." +
        std::to_string(dnn_output->image_msg_header->stamp.nanosec);

    std::shared_ptr<std::vector<hbDNNRoi>> rois = nullptr;
    std::map<size_t, size_t> valid_roi_idx;
    ai_msgs::msg::PerceptionTargets::UniquePtr ai_msg = nullptr;
    if (ai_msg_manage_->GetTargetRois(dnn_output->image_msg_header->stamp,
                                      rois,
                                      valid_roi_idx,
                                      ai_msg,
                                      std::bind(&HandLmkDetNode::NormalizeRoi, this,
                                        std::placeholders::_1, std::placeholders::_2,
                                        expand_scale_, pyramid->width, pyramid->height),
                                      200) < 0 ||
        !ai_msg) {
      RCLCPP_INFO(this->get_logger(),
                  "Frame ts %s get hand roi fail",
                  ts.c_str());
      continue;
    }
    if (!rois || rois->empty() || rois->size() != valid_roi_idx.size()) {
      RCLCPP_INFO(this->get_logger(),
                  "Frame ts %s has no hand roi",
                  ts.c_str());
      if (!rois) {
        rois = std::make_shared<std::vector<hbDNNRoi>>();
      }
    }

    dnn_output->valid_rois = rois;
    dnn_output->valid_roi_idx = valid_roi_idx;
    dnn_output->ai_msg = std::move(ai_msg);

    auto model_manage = GetModel();
    if (!model_manage) {
      RCLCPP_ERROR(this->get_logger(), "Invalid model");
      continue;
    }

    // 2. 使用pyramid创建DNNInput对象inputs
    // inputs将会作为模型的输入通过RunInferTask接口传入
    std::vector<std::shared_ptr<DNNInput>> inputs;
    for (size_t i = 0; i < dnn_output->valid_rois->size(); i++) {
      for (int32_t j = 0; j < model_manage->GetInputCount(); j++) {
        inputs.push_back(pyramid);
      }
    }

    struct timespec time_now = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_now);
    dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
    dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;

    uint32_t ret = 0;
    // 3. 开始预测
    ret = Predict(inputs, dnn_output->valid_rois, dnn_output);

    // 4. 处理预测结果，如渲染到图片或者发布预测结果
    if (ret != 0) {
      continue;
    }
  }
}

int HandLmkDetNode::NormalizeRoi(const hbDNNRoi *src,
                            hbDNNRoi *dst,
                            float norm_ratio,
                            uint32_t total_w,
                            uint32_t total_h) {
  *dst = *src;
  float box_w = dst->right - dst->left;
  float box_h = dst->bottom - dst->top;
  float center_x = (dst->left + dst->right) / 2.0f;
  float center_y = (dst->top + dst->bottom) / 2.0f;
  float w_new = box_w;
  float h_new = box_h;
  
  // {"norm_by_lside_ratio", NormMethod::BPU_MODEL_NORM_BY_LSIDE_RATIO},
  h_new = box_h * norm_ratio;
  w_new = box_w * norm_ratio;
  dst->left = center_x - w_new / 2;
  dst->right = center_x + w_new / 2;
  dst->top = center_y - h_new / 2;
  dst->bottom = center_y + h_new / 2;

  dst->left = dst->left < 0 ? 0.0f : dst->left;
  dst->top = dst->top < 0 ? 0.0f : dst->top;
  dst->right = dst->right > total_w ? total_w : dst->right;
  dst->bottom = dst->bottom > total_h ? total_h : dst->bottom;

  // roi's left and top must be even, right and bottom must be odd
  dst->left += (dst->left % 2 == 0 ? 0 : 1);
  dst->top += (dst->top % 2 == 0 ? 0 : 1);
  dst->right -= (dst->right % 2 == 1 ? 0 : 1);
  dst->bottom -= (dst->bottom % 2 == 1 ? 0 : 1);
 
  int32_t roi_w = dst->right - dst->left;
  int32_t roi_h = dst->bottom - dst->top;
  int32_t max_size = std::max(roi_w, roi_h);
  int32_t min_size = std::min(roi_w, roi_h);

  if (max_size < roi_size_max_ && min_size > roi_size_min_) {
    // check success
    RCLCPP_DEBUG(this->get_logger(),
                  "Valid roi: %d %d %d %d, roi_w: %d, roi_h: %d, "
                  "max_size: %d, min_size: %d",
                  dst->left,
                  dst->top,
                  dst->right,
                  dst->bottom,
                  roi_w,
                  roi_h,
                  max_size,
                  min_size);
    return 0;
  } else {
    RCLCPP_INFO(
        this->get_logger(),
        "Filter roi: %d %d %d %d, max_size: %d, min_size: %d",
        dst->left,
        dst->top,
        dst->right,
        dst->bottom,
        max_size,
        min_size);
    if (max_size >= roi_size_max_) {
      RCLCPP_INFO(
          this->get_logger(),
          "Move far from sensor!");
    } else if (min_size <= roi_size_min_) {
      RCLCPP_INFO(
          this->get_logger(),
          "Move close to sensor!");
    }

    return -1;
  }

  return 0;
}

