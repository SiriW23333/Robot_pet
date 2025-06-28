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

#include "include/gesture_preprocess.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "include/gesture_data.h"
#include "rclcpp/rclcpp.hpp"

namespace inference {
void InputFloat2Int(int8_t* feat_buffer1,
                    int input_quanti_factor,
                    float* BPU_input_data,
                    int N,
                    int H,
                    int W,
                    int C,
                    int frame_input_size) {
  int32_t tmp = 0;
  int index = 0;
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          int offset = n * H * W * C + h * W * C + w * C + c;
          tmp = floor(BPU_input_data[offset] * input_quanti_factor);
          if (tmp > 127) {
            tmp = 127;
          } else if (tmp < -128) {
            tmp = -128;
          }
          feat_buffer1[index] = static_cast<int8_t>(tmp);
          index++;
        }
      }
    }
  }
  if (index != frame_input_size) {
    RCLCPP_ERROR(rclcpp::get_logger("pre process"), "frame_input_size unmatch");
  }
}

int GesturePreProcess::Init(const std::string& json_str) {
  if (!json_str.empty()) {
    // todo 从配置文件中更新配置参数
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

  return 0;
}

int GetHWCIndex(DataType data_type,
                TensorLayout layout,
                int* h_idx,
                int* w_idx,
                int* c_idx) {
  switch (data_type) {
    case inference::IMG_TYPE_BGR:
    case inference::IMG_TYPE_RGB:
    case inference::IMG_TYPE_YUV444:
      *h_idx = 1;
      *w_idx = 2;
      *c_idx = 3;
      break;
    case inference::IMG_TYPE_NV12:
    case inference::IMG_TYPE_Y:
    case inference::IMG_TYPE_NV12_SEPARATE:
      *c_idx = 1;
      *h_idx = 2;
      *w_idx = 3;
      break;
    case inference::TENSOR_TYPE_S4:
    case inference::TENSOR_TYPE_U4:
    case inference::TENSOR_TYPE_U8:
    case inference::TENSOR_TYPE_S8:
    case inference::TENSOR_TYPE_F16:
    case inference::TENSOR_TYPE_S16:
    case inference::TENSOR_TYPE_U16:
    case inference::TENSOR_TYPE_F32:
    case inference::TENSOR_TYPE_S32:
    case inference::TENSOR_TYPE_U32:
    case inference::TENSOR_TYPE_F64:
    case inference::TENSOR_TYPE_S64:
    case inference::TENSOR_TYPE_U64: {
      // Feature's height and width is determined by layout.
      switch (layout) {
        case inference::LAYOUT_NONE:
          return -1;
        case inference::LAYOUT_NHWC:
          *h_idx = 1;
          *w_idx = 2;
          *c_idx = 3;
          break;
        case inference::LAYOUT_NCHW:
          *c_idx = 1;
          *h_idx = 2;
          *w_idx = 3;
          break;
        case inference::LAYOUT_NHWC_4W8C:
          *c_idx = 1;
          *h_idx = 2;
          *w_idx = 3;
          break;
        default:
          return -1;
      }
      break;
    }
    default:
      return -1;
  }
  return 0;
}

int AllocModelTensor(std::shared_ptr<DNNTensor>& tensor_ptr,
                     bool need_alloc = true,
                     int batch = 1) {
  if (tensor_ptr) {
    DNNTensor& tensor = *tensor_ptr;
    hbDNNTensorProperties info = tensor.properties;
    // sysMem
    int h_idx, w_idx, c_idx;
    if (info.tensorType < static_cast<int>(DataType::IMG_TYPE_Y) ||
        info.tensorType > static_cast<int>(DataType::TENSOR_TYPE_MAX) ||
        info.tensorLayout < static_cast<int>(TensorLayout::LAYOUT_NHWC) ||
        info.tensorLayout > static_cast<int>(TensorLayout::LAYOUT_NONE)) {
      std::stringstream ss;
      ss << "Invalid data type. tensorType = " << info.tensorType
         << ", tensorLayout = " << info.tensorLayout;
      RCLCPP_ERROR(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());
      return -1;
    }

    if (0 != GetHWCIndex(static_cast<DataType>(info.tensorType),
                         static_cast<TensorLayout>(info.tensorLayout),
                         &h_idx,
                         &w_idx,
                         &c_idx)) {
      std::stringstream ss;
      ss << "GetHWCIndex Failed. tensorType = " << info.tensorType
         << ", tensorLayout = " << info.tensorLayout;
      RCLCPP_ERROR(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());
      return -1;
    }
    int height = info.validShape.dimensionSize[h_idx];
    int stride = info.alignedShape.dimensionSize[w_idx];

    std::stringstream ss;
    ss << "tensorType: " << info.tensorType
       << ", tensorLayout: " << info.tensorLayout;
    RCLCPP_DEBUG(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());

    switch (info.tensorType) {
      case inference::IMG_TYPE_NV12: {
        // shape.d[h_idx] alignedShape.d[w_idx]
        int y_length = height * stride;
        int uv_length = height / 2 * stride;
        if (need_alloc) {
          hbSysMem mem;
          hbSysAllocCachedMem(&mem, y_length + uv_length);
          tensor.sysMem[0].phyAddr = mem.phyAddr;
          tensor.sysMem[0].virAddr = mem.virAddr;
          tensor.sysMem[0].memSize = mem.memSize;
        }
        break;
      }
      case inference::IMG_TYPE_NV12_SEPARATE: {
        int y_length = height * stride;
        int uv_length = height / 2 * stride;
        if (need_alloc) {
          hbSysMem mem_0, mem_1;
          hbSysAllocCachedMem(&mem_0, y_length);
          hbSysAllocCachedMem(&mem_1, uv_length);
          tensor.sysMem[0].phyAddr = mem_0.phyAddr;
          tensor.sysMem[0].virAddr = mem_0.virAddr;
          tensor.sysMem[0].memSize = mem_0.memSize;
          tensor.sysMem[1].phyAddr = mem_1.phyAddr;
          tensor.sysMem[1].virAddr = mem_1.virAddr;
          tensor.sysMem[1].memSize = mem_1.memSize;
        }
        break;
      }

      case inference::TENSOR_TYPE_S8:
      case inference::TENSOR_TYPE_U8:
      case inference::TENSOR_TYPE_F16:
      case inference::TENSOR_TYPE_S16:
      case inference::TENSOR_TYPE_U16:
      case inference::TENSOR_TYPE_F32:
      case inference::TENSOR_TYPE_S32:
      case inference::TENSOR_TYPE_U32:
      case inference::TENSOR_TYPE_F64:
      case inference::TENSOR_TYPE_S64:
      case inference::TENSOR_TYPE_U64: {
        int elem_size = 1;
        if (info.tensorType >= inference::TENSOR_TYPE_F16 &&
            info.tensorType <= inference::TENSOR_TYPE_U16) {
          elem_size = 2;
        } else if (info.tensorType >= inference::TENSOR_TYPE_F32 &&
                   info.tensorType <= inference::TENSOR_TYPE_U32) {
          elem_size = 4;
        } else if (info.tensorType >= inference::TENSOR_TYPE_F64 &&
                   info.tensorType <= inference::TENSOR_TYPE_U64) {
          elem_size = 8;
        }
        // batch针对roi的输出使用
        int length = elem_size * batch;
        for (int j = 0; j < info.alignedShape.numDimensions; j++) {
          length *= info.alignedShape.dimensionSize[j];
        }

        std::stringstream ss;
        ss << "elem_size: " << elem_size << ", alloc tensor len = " << length;

        // 16对齐
        length = ((length + (16 - 1)) / 16 * 16);
        ss << ", after align tensor len = " << length;
        RCLCPP_DEBUG(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());
        if (need_alloc) {
          hbSysAllocCachedMem(&tensor.sysMem[0], length);
        }
        break;
      }
      default:
        std::stringstream ss;
        ss << "AllocModelTensor not support tensorType: " << info.tensorType;
        RCLCPP_ERROR(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());
        return -1;
    }
  }
  return 0;
}

int AllocModelTensors(std::vector<std::shared_ptr<DNNTensor>>& tensors) {
  int layer_num = tensors.size();
  for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
    if (!tensors.at(layer_idx)) {
      continue;
    }
    AllocModelTensor(tensors.at(layer_idx));
  }
  return 0;
}

int FreeTensor(DNNTensor& tensor) {
  switch (tensor.properties.tensorType) {
    case IMG_TYPE_NV12:
    case TENSOR_TYPE_S4:
    case TENSOR_TYPE_U4:
    case TENSOR_TYPE_S8:
    case TENSOR_TYPE_U8:
    case TENSOR_TYPE_F16:
    case TENSOR_TYPE_S16:
    case TENSOR_TYPE_U16:
    case TENSOR_TYPE_F32:
    case TENSOR_TYPE_S32:
    case TENSOR_TYPE_U32:
    case TENSOR_TYPE_F64:
    case TENSOR_TYPE_S64:
    case TENSOR_TYPE_U64: {
      hbSysMem mem = {tensor.sysMem[0].phyAddr,
                      tensor.sysMem[0].virAddr,
                      tensor.sysMem[0].memSize};
      hbSysFreeMem(&mem);
      break;
    }
    case IMG_TYPE_NV12_SEPARATE: {
      hbSysMem mem_0 = {tensor.sysMem[0].phyAddr,
                        tensor.sysMem[0].virAddr,
                        tensor.sysMem[0].memSize};
      hbSysMem mem_1 = {tensor.sysMem[1].phyAddr,
                        tensor.sysMem[1].virAddr,
                        tensor.sysMem[1].memSize};
      hbSysFreeMem(&mem_0);
      hbSysFreeMem(&mem_1);
      break;
    }
    default:
      std::stringstream ss;
      ss << "FreeTensor not support tensorType: "
         << tensor.properties.tensorType;
      RCLCPP_ERROR(rclcpp::get_logger("pre process"), "%s", ss.str().c_str());
      break;
  }
  return 0;
}

int FreeTensors(std::vector<DNNTensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); i++) {
    DNNTensor& tensor = tensors[i];
    FreeTensor(tensor);
  }
  tensors.clear();
  return 0;
}

// 仅赋值TensorProperties，不AllocTensor空间
int PrepareModelTensorProperties(
    const std::vector<hbDNNTensorProperties>& model_info,
    std::vector<DNNTensor>& tensors) {
  int layer_num = model_info.size();
  tensors.resize(layer_num);
  for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
    tensors[layer_idx].properties = model_info[layer_idx];
  }
  return 0;
}

int GesturePreProcess::Execute(
    const ai_msgs::msg::PerceptionTargets::ConstSharedPtr ai_msg,
    const std::vector<hbDNNTensorProperties>& input_model_info_,
    std::vector<std::shared_ptr<DNNTensor>>& input_tensors,
    std::vector<uint64_t>& track_ids,
    uint64_t& ts) {
  // 防止越界
  // static std::once_flag flag;
  // std::call_once(flag, [this, &ai_msg] {
  //   startup_ts_second_ = ai_msg->header.stamp.sec;
  // });
  // 换算成ms
  uint64_t timestamp = (ai_msg->header.stamp.sec - startup_ts_second_) * 1000 +
                       ai_msg->header.stamp.nanosec / 1000000;
  ts = timestamp;

  std::vector<uint64_t> disappeared_hand_ids;
  for (const auto& target : ai_msg->disappeared_targets) {
    for (const auto& roi : target.rois) {
      if ("hand" == roi.type) {
        disappeared_hand_ids.push_back(target.track_id);
      }
    }
  }

  if (!ai_msg->targets.empty()) {
    for (const auto& target : ai_msg->targets) {
      uint64_t track_id = target.track_id;
      std::vector<std::shared_ptr<hbDNNRoi>> rois;
      std::vector<std::shared_ptr<inference::Landmarks>> lmkses;

      RCLCPP_DEBUG(rclcpp::get_logger("preprocess"),
                   "target rois size: %ld, points size: %ld",
                   target.rois.size(),
                   target.points.size());

      for (const auto& roi : target.rois) {
        RCLCPP_DEBUG(
            rclcpp::get_logger("preprocess"), "roi.type: %s", roi.type.c_str());
        if ("hand" == roi.type) {
          auto dnn_roi = std::make_shared<hbDNNRoi>();
          dnn_roi->left = roi.rect.x_offset;
          dnn_roi->top = roi.rect.y_offset;
          dnn_roi->right = roi.rect.x_offset + roi.rect.width;
          dnn_roi->bottom = roi.rect.y_offset + roi.rect.height;
          rois.push_back(dnn_roi);
        }
      }
      if (rois.empty()) {
        continue;
      }

      RCLCPP_INFO(rclcpp::get_logger("preprocess"),
                  "target id: %ld has hand roi size: %ld",
                  track_id,
                  rois.size());
      for (const auto& point : target.points) {
        RCLCPP_DEBUG(rclcpp::get_logger("preprocess"),
                     "point.type: %s,  size: %ld",
                     point.type.c_str(),
                     point.point.size());
        auto lmk = std::make_shared<inference::Landmarks>();
        if ("hand_kps" == point.type) {
          for (const auto& pt : point.point) {
            lmk->push_back(inference::HobotPoint(pt.x, pt.y));
          }
        }
        RCLCPP_DEBUG(rclcpp::get_logger("preprocess"),
                     "hand lmk point size: %ld",
                     lmk->size());
        lmkses.push_back(lmk);
      }

      RCLCPP_INFO(
          rclcpp::get_logger("preprocess"), "hand lmk size: %ld", lmkses.size());

      if (rois.size() != lmkses.size()) {
        RCLCPP_INFO(
            rclcpp::get_logger("preprocess"),
            "target id: %ld rois.size: %ld is unmatch with lmkses.size: %ld",
            track_id,
            rois.size(),
            lmkses.size());
        continue;
      }

      if (rois.empty()) {
        return 0;
      }

      for (size_t roi_idx = 0; roi_idx < rois.size(); ++roi_idx) {
        auto roi = rois[roi_idx];
        auto lmks = lmkses[roi_idx];

        int8_t* feature_buf = nullptr;
        static int input_size = 1;

        // preprocess
        auto cached_kpses = std::make_shared<
            std::vector<std::shared_ptr<inference::Landmarks>>>();
        // currently only support gesture detection
        RCLCPP_DEBUG(
            rclcpp::get_logger("preprocess"), "in kps->size: %ld", lmks->size());
        lmks_proc_.Execute(track_id, roi, lmks, cached_kpses, timestamp);
        if (cached_kpses->size() < static_cast<uint32_t>(seq_len_)) {
          continue;
        }

        std::shared_ptr<DNNTensor> input_tensor_ptr =
            std::shared_ptr<DNNTensor>(new DNNTensor(),
                                       [this](DNNTensor* tensors_ptr) {
                                         if (tensors_ptr) {
                                           FreeTensor(*tensors_ptr);
                                           delete tensors_ptr;
                                           tensors_ptr = nullptr;
                                         }
                                       });

        // prepare tensor properties
        std::vector<DNNTensor> tensors;
        PrepareModelTensorProperties(input_model_info_, tensors);
        *input_tensor_ptr = tensors.front();
        static std::vector<int> input_valid_shape;
        static std::vector<int> input_aligned_shape;
        static std::once_flag flag;
        std::call_once(flag, [&input_tensor_ptr] {
          for (int i = 0;
               i < input_tensor_ptr->properties.validShape.numDimensions;
               ++i) {
            input_valid_shape.push_back(
                input_tensor_ptr->properties.validShape.dimensionSize[i]);
          }
          for (int i = 0;
               i < input_tensor_ptr->properties.alignedShape.numDimensions;
               ++i) {
            input_aligned_shape.push_back(
                input_tensor_ptr->properties.alignedShape.dimensionSize[i]);
            input_size *= input_aligned_shape[i];
          }
        });

        RCLCPP_DEBUG(rclcpp::get_logger("prepro"),
                     "input_aligned_shape: %d %d %d %d",
                     input_aligned_shape[0],
                     input_aligned_shape[1],
                     input_aligned_shape[2],
                     input_aligned_shape[3]);
        RCLCPP_DEBUG(rclcpp::get_logger("prepro"),
                     "input_valid_shape: %d %d %d %d",
                     input_valid_shape[0],
                     input_valid_shape[1],
                     input_valid_shape[2],
                     input_valid_shape[3]);

        // tensor with aligned shape, padding 0
        LmksProcessTensor tensor(input_aligned_shape[0],
                                 input_aligned_shape[1],
                                 input_aligned_shape[2],
                                 input_aligned_shape[3]);
        for (int nn = 0; nn < input_valid_shape[0]; ++nn) {
          for (int hh = 0; hh < input_valid_shape[1]; ++hh) {
            for (int ww = 0; ww < input_valid_shape[2]; ++ww) {
              int cached_data_idx = input_valid_shape[2] == seq_len_ ? ww : hh;
              int kps_idx = input_valid_shape[1] == kps_len_ ? hh : ww;
              auto cur_kps = cached_kpses->at(cached_data_idx);
              tensor.Set(nn, hh, ww, 0, cur_kps->at(kps_idx).x);
              tensor.Set(nn, hh, ww, 1, cur_kps->at(kps_idx).y);
              tensor.Set(nn, hh, ww, 2, cur_kps->at(kps_idx).score);
            }
          }
        }

        float* BPU_input_data = tensor.data.data();
        feature_buf = static_cast<int8_t*>(malloc(input_size));
        int input_quanti_factor = 1 << input_shift_;
        InputFloat2Int(feature_buf,
                       input_quanti_factor,
                       BPU_input_data,
                       input_aligned_shape[0],
                       input_aligned_shape[1],
                       input_aligned_shape[2],
                       input_aligned_shape[3],
                       input_size);

        // 申请Tensor
        AllocModelTensor(input_tensor_ptr);
        if (input_tensor_ptr->properties.tensorType ==
            inference::TENSOR_TYPE_S8) {
          if (static_cast<int>(input_tensor_ptr->sysMem[0].memSize) !=
              input_size) {
            RCLCPP_ERROR(rclcpp::get_logger("pre pro"),
                         "allocated input size: %d need: %d",
                         input_tensor_ptr->sysMem[0].memSize,
                         input_size);
            return -1;
          }
          memcpy(input_tensor_ptr->sysMem[0].virAddr,
                 reinterpret_cast<uint8_t*>(feature_buf),
                 input_tensor_ptr->sysMem[0].memSize);
        } else {
          RCLCPP_ERROR(
              rclcpp::get_logger("pre pro"),
              "Execute not support data type: %d",
              static_cast<int>(input_tensor_ptr->properties.tensorType));
          return -1;
        }

        input_tensors.emplace_back(input_tensor_ptr);
        track_ids.push_back(track_id);

        if (feature_buf) {
          free(feature_buf);
          feature_buf = nullptr;
        }
      }
    }
  }

  lmks_proc_.Clean(disappeared_hand_ids);
  // 用于没有mot场景下的缓存清理
  // 或者没有订阅到消失的ID
  lmks_proc_.Clean(timestamp);

  return 0;
}
}  // namespace inference
