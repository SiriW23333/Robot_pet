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

#include "include/hand_lmk_output_parser.h"

int32_t HandLmkOutputParser::Parse(
    std::shared_ptr<LandmarksResult> &output,
    std::shared_ptr<DNNTensor> &output_tensor,
    std::shared_ptr<std::vector<hbDNNRoi>> rois) {

  if (rois == nullptr || static_cast<int>(rois->size()) == 0) {
    RCLCPP_INFO(rclcpp::get_logger("hand lmk parser"), "get null rois");
    return -1;
  }

  std::shared_ptr<LandmarksResult> landmarks_result = nullptr;
  if (!output) {
    landmarks_result = std::make_shared<LandmarksResult>();
    landmarks_result->Reset();
    output = landmarks_result;
  } else {
    landmarks_result = std::dynamic_pointer_cast<LandmarksResult>(output);
    landmarks_result->Reset();
  }

  auto out_layers = 1;  // model_output_count_;
  // 定点转浮点
  std::vector<FloatTensor> float_tensors;
  float_tensors.resize(out_layers);

  if (!output_tensor) {
    RCLCPP_ERROR(rclcpp::get_logger("hand lmk parser"), "invalid out tensor");
    return -1;
  }
  int batch = 1;  // rois->size();
  for (int i = 0; i < out_layers; i++) {
    const DNNTensor &tensor = *output_tensor;  // task->output_tensors_[i];
    FloatTensor &float_tensor = float_tensors[i];
    OutputTensors2FloatTensors(tensor, float_tensor, batch);
  }

  // 对应输出层每个box_result的有效大小
  static int valid_offset = 1;
  static std::once_flag flag;
  std::call_once(flag, [&output_tensor]() {
    for (int dim_idx = 1; dim_idx < 4; dim_idx++) {
      valid_offset *=
          output_tensor->properties.validShape.dimensionSize[dim_idx];
    }
  });

  // 取对应的float_tensor解析
  for (int roi_idx = 0; roi_idx < static_cast<int>(rois->size()); roi_idx++) {
    LmksPostPro(float_tensors[0],
                  valid_offset,
                  roi_idx,
                  rois->at(roi_idx),
                  landmarks_result);
  }

  return 0;
}

void HandLmkOutputParser::LmksPostPro(
    const FloatTensor &float_tensor,
    const int valid_offset,
    const int valid_result_idx,
    const hbDNNRoi &roi,
    std::shared_ptr<LandmarksResult> &output) {
  auto mxnet_output =
      float_tensor.value.data() + valid_result_idx * valid_offset;
  int box_height = floor(roi.bottom - roi.top);
  int box_width = floor(roi.right - roi.left);
  auto ratio_h = float_tensor.dim[1] * i_o_stride_;
  auto ratio_w = float_tensor.dim[2] * i_o_stride_;

  int step = 4;  // 4 for performance optimization, use 1 for normal cases
  Landmarks landmarks;
  landmarks.resize(float_tensor.dim[3]);

  std::stringstream ss;
  ss << "hand_lmk:\n";
  for (int c = 0; c < float_tensor.dim[3]; ++c) {  // c
    float max_value = 0;
    int max_index[2] = {0, 0};
    for (auto h = 0; h < float_tensor.dim[1]; h += step) {    // h
      for (auto w = 0; w < float_tensor.dim[2]; w += step) {  // w
        int index = h * float_tensor.dim[2] * float_tensor.dim[3] +
                    w * float_tensor.dim[3] + c;
        float value = mxnet_output[index];
        if (value > max_value) {
          max_value = value;
          max_index[0] = h;
          max_index[1] = w;
        }
      }  // w
    }    // h
    // performance optimization
    auto is_max = false;
    auto campare_func =
        [&float_tensor, &max_index, &max_value, &is_max, mxnet_output, this](
            int h, int w, int c) {
          int index = h * float_tensor.dim[2] * float_tensor.dim[3] +
                      w * float_tensor.dim[3] + c;
          if (max_value < mxnet_output[index]) {
            max_value = mxnet_output[index];
            max_index[0] = h;
            max_index[1] = w;
            is_max = false;
          }
        };
    while (false == is_max) {
      is_max = true;
      int h = max_index[0];
      int w = max_index[1];
      if (h > 0) {
        campare_func(h - 1, w, c);
      }
      if (h < float_tensor.dim[1] - 1) {
        campare_func(h + 1, w, c);
      }
      if (w > 0) {
        campare_func(h, w - 1, c);
      }
      if (w < float_tensor.dim[2] - 1) {
        campare_func(h, w + 1, c);
      }
    }
    // end performance optimization

    float y = max_index[0];
    float x = max_index[1];

    float diff_x = 0, diff_y = 0;
    if (y > 0 && y < float_tensor.dim[1] - 1) {
      int top = (y - 1) * float_tensor.dim[2] * float_tensor.dim[3] +
                x * float_tensor.dim[3] + c;
      int down = (y + 1) * float_tensor.dim[2] * float_tensor.dim[3] +
                 x * float_tensor.dim[3] + c;
      diff_y = mxnet_output[down] - mxnet_output[top];
    }
    if (x > 0 && x < float_tensor.dim[2] - 1) {
      int left = y * float_tensor.dim[2] * float_tensor.dim[3] +
                 (x - 1) * float_tensor.dim[3] + c;
      int right = y * float_tensor.dim[2] * float_tensor.dim[3] +
                  (x + 1) * float_tensor.dim[3] + c;
      diff_x = mxnet_output[right] - mxnet_output[left];
    }

    // y = y + (diff_y * diff_coeff + 0.5); // for float32 of model output
    // x = x + (diff_x * diff_coeff + 0.5); // for float32 of model output
    y = y + (diff_y > 0 ? 0.25 : -0.25) + 0.5;
    x = x + (diff_x > 0 ? 0.25 : -0.25) + 0.5;
    y = y * i_o_stride_;
    x = x * i_o_stride_;

    y = y * box_height / ratio_h + roi.top;
    x = x * box_width / ratio_w + roi.left;

    auto &point = landmarks[c];
    point.x = x;
    point.y = y;
    point.score = max_value;
    ss << c << " x y score:( " << x << " " << y << " " << point.score << ")\n";
  }  // c
  RCLCPP_DEBUG(rclcpp::get_logger("hand lmk parser"), "%s", ss.str().c_str());
  output->values.emplace_back(landmarks);
}

void convert(int8_t *&cur_c_dst, int8_t tmp_int8_value, uint8_t shift) {
  float tmp_float_value;
  tmp_float_value =
      (static_cast<float>(tmp_int8_value)) / (static_cast<float>(1 << shift));
  *(reinterpret_cast<float *>(cur_c_dst)) = tmp_float_value;
}

void cal_float_tensor_dim3(const DNNTensor &tensor,
                           FloatTensor &float_tensor,
                           int hh,
                           void *&cur_w_dst,
                           void *&cur_w_src,
                           int src_elem_size,
                           int dst_elem_size) {
  for (int cc = 0; cc < float_tensor.dim[3]; cc++) {
    int8_t *cur_c_dst =
        reinterpret_cast<int8_t *>(cur_w_dst) + cc * dst_elem_size;
    int8_t *cur_c_src =
        reinterpret_cast<int8_t *>(cur_w_src) + cc * src_elem_size;
    if (cur_c_src == NULL || cur_c_dst == NULL) {
      RCLCPP_ERROR(rclcpp::get_logger("post process"), "line:%d", __LINE__);
      return;
    }
    uint8_t shift = 0;
    if (float_tensor.layout == LAYOUT_NHWC) {
      shift = tensor.properties.shift.shiftData[cc];
    } else if (float_tensor.layout == LAYOUT_NCHW) {
      shift = tensor.properties.shift.shiftData[hh];
    }

    int8_t tmp_int8_value = *cur_c_src;
    float tmp_float_value = 0;
    uint64_t tmp_int_shift = (1 << shift);
    float tmp_float_shift = static_cast<float>(tmp_int_shift);
    if (tmp_float_shift != 0) {
      tmp_float_value = (static_cast<float>(tmp_int8_value)) / tmp_float_shift;
    }
    float *float_cur_c_dst = reinterpret_cast<float *>(cur_c_dst);
    *float_cur_c_dst = tmp_float_value;
  }
}

void HandLmkOutputParser::OutputTensors2FloatTensors(const DNNTensor &tensor,
                                                     FloatTensor &float_tensor,
                                                     int batch) {
  auto tensor_type = static_cast<DataType>(tensor.properties.tensorType);
  switch (tensor_type) {
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

    case TENSOR_TYPE_S8: {
      int src_elem_size = 1;
      int dst_elem_size = 4;  // float
      if (tensor_type == TENSOR_TYPE_S32 || tensor_type == TENSOR_TYPE_U32) {
        src_elem_size = 4;
      } else if (tensor_type == TENSOR_TYPE_S64) {
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
      RCLCPP_ERROR(rclcpp::get_logger("hand lmk parser"),
                   "not support tensorType: %d",
                   static_cast<int>(tensor_type));
  }
}
