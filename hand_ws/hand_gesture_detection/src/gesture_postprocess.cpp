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

#include "include/gesture_postprocess.h"

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace inference {
void cal_float_tensor_dim3(const DNNTensor &tensor,
                           FloatTensor &float_tensor,
                           int hh,
                           void *cur_w_dst,
                           void *cur_w_src,
                           int src_elem_size,
                           int dst_elem_size) {
  // val is 4
  RCLCPP_DEBUG(
      rclcpp::get_logger("post process"), "src_elem_size: %d", src_elem_size);
  for (int cc = 0; cc < float_tensor.dim[3]; cc++) {
    void *cur_c_dst =
        reinterpret_cast<int8_t *>(cur_w_dst) + cc * dst_elem_size;
    void *cur_c_src =
        reinterpret_cast<int8_t *>(cur_w_src) + cc * src_elem_size;

    if (src_elem_size == 4) {
      int32_t tmp_int32_value = *(reinterpret_cast<int32_t *>(cur_c_src));
      // 转浮点
      float tmp_float_value = tmp_int32_value;
      // 确定对应的shift
      uint8_t shift = 0;
      if (float_tensor.layout == LAYOUT_NHWC) {
        shift = tensor.properties.shift.shiftData[cc];
      } else if (float_tensor.layout == LAYOUT_NCHW) {
        shift = tensor.properties.shift.shiftData[hh];
      }

      if (tmp_int32_value != 0) {
        int *ix = reinterpret_cast<int *>(&tmp_float_value);
        (*ix) -= shift * 0x00800000;
      }
      *(reinterpret_cast<float *>(cur_c_dst)) = tmp_float_value;

    } else if (src_elem_size == 1) {
      uint8_t shift = 0;
      if (float_tensor.layout == LAYOUT_NHWC) {
        shift = tensor.properties.shift.shiftData[cc];
      } else if (float_tensor.layout == LAYOUT_NCHW) {
        shift = tensor.properties.shift.shiftData[hh];
      }
      *(reinterpret_cast<float *>(cur_c_dst)) =
          (static_cast<float>(*(reinterpret_cast<int8_t *>(cur_c_src)))) /
          (static_cast<float>(1 << shift));
    } else {  // src_elem_size == 8
      int64_t tmp_int64_value = *(reinterpret_cast<int64_t *>(cur_c_src));
      // 转浮点
      float tmp_float_value;
      // 确定对应的shift
      uint8_t shift = 0;
      if (float_tensor.layout == LAYOUT_NHWC) {
        shift = tensor.properties.shift.shiftData[cc];
      } else if (float_tensor.layout == LAYOUT_NCHW) {
        shift = tensor.properties.shift.shiftData[hh];
      }
      tmp_float_value = (static_cast<float>(tmp_int64_value)) /
                        (static_cast<float>(1 << shift));
      *(reinterpret_cast<float *>(cur_c_dst)) = tmp_float_value;
    }
  }
}

void GesturePostProcess::OutputTensors2FloatTensors(const DNNTensor &tensor,
                                                    FloatTensor &float_tensor,
                                                    int batch) {
  auto tensorType = static_cast<DataType>(tensor.properties.tensorType);
  switch (tensorType) {
    // 模型输出直接是float，直接复制
    case TENSOR_TYPE_F32: {
      // float_tensor.layout = tensor.properties.tensorLayout;
      int elem_size = 4;  // float32
      int batch_valid_size = 1;
      int batch_aligned_size = 1;
      for (int i = 0; i < 4; i++) {
        float_tensor.dim[i] = tensor.properties.validShape.dimensionSize[i];
        batch_valid_size *= float_tensor.dim[i];
        batch_aligned_size *= tensor.properties.alignedShape.dimensionSize[i];
      }
      for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        // void *dst = float_tensor.value.data();
        void *dst = &float_tensor.value[batch_idx * batch_valid_size];
        void *src = reinterpret_cast<int8_t *>(tensor.sysMem[0].virAddr) +
                    batch_idx * batch_aligned_size * elem_size;

        uint32_t dst_n_stride = float_tensor.dim[1] * float_tensor.dim[2] *
                                float_tensor.dim[3] * elem_size;
        uint32_t dst_h_stride =
            float_tensor.dim[2] * float_tensor.dim[3] * elem_size;
        uint32_t dst_w_stride = float_tensor.dim[3] * elem_size;
        uint32_t src_n_stride = tensor.properties.validShape.dimensionSize[1] *
                                tensor.properties.validShape.dimensionSize[2] *
                                tensor.properties.validShape.dimensionSize[3] *
                                elem_size;
        uint32_t src_h_stride = tensor.properties.validShape.dimensionSize[2] *
                                tensor.properties.validShape.dimensionSize[3] *
                                elem_size;
        uint32_t src_w_stride =
            tensor.properties.validShape.dimensionSize[3] * elem_size;
        for (int nn = 0; nn < float_tensor.dim[0]; nn++) {
          void *cur_n_dst = reinterpret_cast<int8_t *>(dst) + nn * dst_n_stride;
          void *cur_n_src = reinterpret_cast<int8_t *>(src) + nn * src_n_stride;
          for (int hh = 0; hh < float_tensor.dim[1]; hh++) {
            void *cur_h_dst =
                reinterpret_cast<int8_t *>(cur_n_dst) + hh * dst_h_stride;
            void *cur_h_src =
                reinterpret_cast<int8_t *>(cur_n_src) + hh * src_h_stride;
            for (int ww = 0; ww < float_tensor.dim[2]; ww++) {
              void *cur_w_dst =
                  reinterpret_cast<int8_t *>(cur_h_dst) + ww * dst_w_stride;
              void *cur_w_src =
                  reinterpret_cast<int8_t *>(cur_h_src) + ww * src_w_stride;
              memcpy(cur_w_dst, cur_w_src, float_tensor.dim[3] * elem_size);
            }
          }
        }
      }
      break;
    }
    case TENSOR_TYPE_S8:
    case TENSOR_TYPE_U8:
    case TENSOR_TYPE_S32:
    case TENSOR_TYPE_U32:
    case TENSOR_TYPE_S64: {
      int src_elem_size = 1;
      int dst_elem_size = 4;  // float
      if (tensorType == TENSOR_TYPE_S32 || tensorType == TENSOR_TYPE_U32) {
        src_elem_size = 4;
      } else if (tensorType == TENSOR_TYPE_S64) {
        src_elem_size = 8;
      }
      // convert to float
      float_tensor.layout =
          static_cast<TensorLayout>(tensor.properties.tensorLayout);
      int batch_valid_size = 1;
      int batch_aligned_size = 1;
      for (int i = 0; i < 4; i++) {
        float_tensor.dim[i] = tensor.properties.validShape.dimensionSize[i];
        batch_valid_size *= float_tensor.dim[i];
        batch_aligned_size *= tensor.properties.alignedShape.dimensionSize[i];
      }
      float_tensor.value.resize(batch_valid_size * batch);
      for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        void *dst = &float_tensor.value[batch_idx * batch_valid_size];
        void *src = reinterpret_cast<int8_t *>(tensor.sysMem[0].virAddr) +
                    batch_idx * batch_aligned_size * src_elem_size;
        uint32_t dst_n_stride = float_tensor.dim[1] * float_tensor.dim[2] *
                                float_tensor.dim[3] * dst_elem_size;
        uint32_t dst_h_stride =
            float_tensor.dim[2] * float_tensor.dim[3] * dst_elem_size;
        uint32_t dst_w_stride = float_tensor.dim[3] * dst_elem_size;
        uint32_t src_n_stride =
            tensor.properties.alignedShape.dimensionSize[1] *
            tensor.properties.alignedShape.dimensionSize[2] *
            tensor.properties.alignedShape.dimensionSize[3] * src_elem_size;
        uint32_t src_h_stride =
            tensor.properties.alignedShape.dimensionSize[2] *
            tensor.properties.alignedShape.dimensionSize[3] * src_elem_size;
        uint32_t src_w_stride =
            tensor.properties.alignedShape.dimensionSize[3] * src_elem_size;

        for (int nn = 0; nn < float_tensor.dim[0]; nn++) {
          void *cur_n_dst = reinterpret_cast<int8_t *>(dst) + nn * dst_n_stride;
          void *cur_n_src = reinterpret_cast<int8_t *>(src) + nn * src_n_stride;
          for (int hh = 0; hh < float_tensor.dim[1]; hh++) {
            void *cur_h_dst =
                reinterpret_cast<int8_t *>(cur_n_dst) + hh * dst_h_stride;
            void *cur_h_src =
                reinterpret_cast<int8_t *>(cur_n_src) + hh * src_h_stride;
            for (int ww = 0; ww < float_tensor.dim[2]; ww++) {
              void *cur_w_dst =
                  reinterpret_cast<int8_t *>(cur_h_dst) + ww * dst_w_stride;
              void *cur_w_src =
                  reinterpret_cast<int8_t *>(cur_h_src) + ww * src_w_stride;
              cal_float_tensor_dim3(tensor,
                                    float_tensor,
                                    hh,
                                    cur_w_dst,
                                    cur_w_src,
                                    src_elem_size,
                                    dst_elem_size);
            }
          }
        }
      }
      break;
    }
    default:
      RCLCPP_ERROR(rclcpp::get_logger("post process"),
                   "not support tensorType: %d",
                   static_cast<int>(tensorType));
  }
}

void GesturePostProcess::InitThresholds(std::string &thresholds_str) {
  size_t delimiter = thresholds_str.find(";");
  std::string group = "";
  while (delimiter != std::string::npos) {
    // one group
    group = thresholds_str.substr(0, delimiter);
    // remove current group from groups
    thresholds_str = thresholds_str.substr(delimiter + 1,
                                           thresholds_str.length() - delimiter);
    // string to float
    thresholds_.push_back(std::stof(group));
    delimiter = thresholds_str.find(";");
  }
  if (thresholds_str.length() != 0) {  // only one group
    thresholds_.push_back(std::stof(thresholds_str));
  }

  std::stringstream ss;
  ss << "thresholds_.size():" << thresholds_.size();
  for (const auto &thr : thresholds_) {
    ss << "thr:" << thr;
  }
  RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());
}

void GesturePostProcess::InitMergeGroups(std::string &merge_groups_str) {
  size_t delimiter = merge_groups_str.find(";");
  size_t sub_delimiter = std::string::npos;
  std::string group = "";
  std::vector<int> vec;
  while (delimiter != std::string::npos) {
    // one group
    group = merge_groups_str.substr(0, delimiter);
    // remove brackets
    group = group.substr(1, group.length() - 2);
    // remove current group from groups
    merge_groups_str = merge_groups_str.substr(
        delimiter + 1, merge_groups_str.length() - delimiter);
    // string to int
    sub_delimiter = group.find(",");
    while (sub_delimiter != std::string::npos) {
      int index = std::stoi(group.substr(0, sub_delimiter));
      vec.push_back(index);
      group = group.substr(sub_delimiter + 1, group.length() - sub_delimiter);
      sub_delimiter = group.find(",");
    }
    // [] without number is placeholder
    if (!group.empty()) {
      int index = std::stoi(group);
      vec.push_back(index);
    }
    merge_groups_.push_back(vec);
    delimiter = merge_groups_str.find(";");
    vec.clear();
  }
  if (merge_groups_str.length() != 0) {  // only one group
    group = merge_groups_str.substr(1, merge_groups_str.length() - 2);
    sub_delimiter = group.find(",");
    while (sub_delimiter != std::string::npos) {
      int index = std::stoi(group.substr(0, sub_delimiter));
      vec.push_back(index);
      group = group.substr(sub_delimiter + 1, group.length() - sub_delimiter);
      sub_delimiter = group.find(",");
    }
    if (!group.empty()) {
      int index = std::stoi(group);
      vec.push_back(index);
    }
    merge_groups_.push_back(vec);
    vec.clear();
  }
}

int GesturePostProcess::Init(const std::string &json_str) {
  if (!json_str.empty()) {
    // todo 从配置文件中更新配置参数
  }
  std::stringstream ss;
  ss << "threshold: " << threshold_ << ", en_score_avg: " << en_score_avg_;
  RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());

  return 0;
}

std::shared_ptr<GestureRes> GesturePostProcess::Execute(
    const std::shared_ptr<DNNTensor> &output_tensor,
    const uint64_t &track_id,
    const uint64_t &timestamp) {
  auto act_results = std::make_shared<GestureRes>();
  // each task corresponds to one target
  auto out_layers = 1;  // model_output_count_;
  // auto &properties = output_tensor->properties;
  // 定点转浮点
  std::vector<FloatTensor> float_tensors_;
  float_tensors_.resize(out_layers);
  int batch = out_layers;
  const DNNTensor &tensor = *output_tensor;  // task->output_tensors_[i];
  FloatTensor &float_tensor = float_tensors_[0];
  OutputTensors2FloatTensors(tensor, float_tensor, batch);
  if (!float_tensors_.empty()) {
    *act_results = ActPostPro(float_tensors_, timestamp, track_id);
  } else {
    act_results->value_ = -1;
    act_results->score_ = 0.0f;
    act_results->state_ = -1;
  }

  return act_results;
}

GestureRes GesturePostProcess::ActPostPro(
    std::vector<FloatTensor> &float_tensors,
    uint64_t timestamp,
    uint64_t track_id) {
  GestureRes act_result;
  if (float_tensors.empty()) {
    std::stringstream ss;
    ss << "no bpu output for this target, track_id: " << track_id
       << ", set INVALID";
    RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());
    act_result.value_ = -1;
    act_result.score_ = 0.0f;
    act_result.state_ = -1;
  } else {
    auto mxnet_output = float_tensors[0].value.data();
    int output_size = float_tensors[0].dim[0] * float_tensors[0].dim[1] *
                      float_tensors[0].dim[2] * float_tensors[0].dim[3];
    std::vector<float> model_outs;
    model_outs.resize(output_size);
    float max_score = mxnet_output[0], sum_score = 0;
    // sofmax
    for (int i = 0; i < output_size; ++i) {
      model_outs[i] = mxnet_output[i];
      if (mxnet_output[i] > max_score) {
        max_score = mxnet_output[i];
      }
    }
    for (auto &item : model_outs) {
      item = std::exp(item - max_score);
      sum_score += item;
    }

    // merge by group
    std::vector<float> act_rets(merge_groups_.size(), 0);
    float max_group_score = 0;
    int max_group_index = -1;
    if (en_score_avg_) {
      std::vector<float> tmp_rets(merge_groups_.size(), 0);
      for (size_t g_idx = 0; g_idx < merge_groups_.size(); ++g_idx) {
        for (size_t idx = 0; idx < merge_groups_[g_idx].size(); ++idx) {
          tmp_rets[g_idx] += model_outs[merge_groups_[g_idx][idx]] / sum_score;
        }
      }

      auto avg_rets = GesturePostProcessUtil::GetInstance()->GetCachedAvgScore(
          static_cast<float>(timestamp / 1000), track_id, tmp_rets);

      for (size_t idx = 0; idx < avg_rets.size(); ++idx) {
        if (avg_rets[idx] > max_group_score) {
          max_group_score = avg_rets[idx];
          max_group_index = idx;
        }
      }
    } else {
      for (size_t g_idx = 0; g_idx < merge_groups_.size(); ++g_idx) {
        for (size_t idx = 0; idx < merge_groups_[g_idx].size(); ++idx) {
          act_rets[g_idx] += model_outs[merge_groups_[g_idx][idx]] / sum_score;
        }
        if (act_rets[g_idx] > max_group_score) {
          max_group_score = act_rets[g_idx];
          max_group_index = g_idx;
        }
      }
    }

    // currently only support gesture
    float threshold = threshold_;
    if (max_group_index >= 0 &&
        max_group_index < static_cast<int>(thresholds_.size())) {
      threshold = thresholds_.at(max_group_index);
    }
    if (max_group_score >= threshold) {
      act_result.value_ = max_group_index;
      act_result.score_ = max_group_score;
    } else {
      act_result.value_ = 0;
      act_result.score_ = max_group_score;
    }
    std::stringstream ss;
    ss << "track_id: " << track_id << ", act val: " << act_result.value_
       << ", score: " << act_result.score_ << ", threshold: " << threshold
       << ", max_group_index: " << max_group_index
       << ", max_group_score: " << max_group_score;
    RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());
  }
  return act_result;
}

}  // namespace inference
