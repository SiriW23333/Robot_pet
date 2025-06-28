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

#ifndef HAND_LMK_OUTPUT_PARSER_H
#define HAND_LMK_OUTPUT_PARSER_H

#include <memory>
#include <string>
#include <utility>
#include <vector>


#include "rclcpp/rclcpp.hpp"

#include "dnn_node/dnn_node_data.h"


// using hobot::dnn_node::DNNResult;
using hobot::dnn_node::DNNTensor;
// using hobot::dnn_node::InputDescription;
using hobot::dnn_node::Model;
// using hobot::dnn_node::OutputDescription;
// using hobot::dnn_node::SingleBranchOutputParser;

/**
 * \~Chinese @brief 2D坐标点
 */
template <typename Dtype>
struct Point_ {
  inline Point_() {}
  inline Point_(Dtype x_, Dtype y_, float score_ = 0.0)
      : x(x_), y(y_), score(score_) {}

  Dtype x = 0;
  Dtype y = 0;
  float score = 0.0;
};
typedef Point_<float> Point;
typedef std::vector<Point> Landmarks;

class LandmarksResult {
 public:
  std::vector<Landmarks> values;

  void Reset() { values.clear(); }
};

// class HandLmkOutDesc : public OutputDescription {
//  public:
//   HandLmkOutDesc(Model* mode, int index, std::string type = "detection")
//       : OutputDescription(mode, index, type) {}

//   std::shared_ptr<std::vector<hbDNNRoi>> rois;
//   std::string ts;
//   // 由于每个roi对应一次Parse，使用present_roi_idx统计当前Parse对应的roi idx
//   size_t present_roi_idx = 0;
// };

typedef enum {
  LAYOUT_NHWC = 0,
  LAYOUT_NCHW = 2,
  LAYOUT_NHWC_4W8C = 134,  // 适配老模型中的layout特殊处理
  LAYOUT_NONE = 255,
} TensorLayout;

typedef enum {
  IMG_TYPE_Y,
  IMG_TYPE_NV12,
  IMG_TYPE_NV12_SEPARATE,
  IMG_TYPE_YUV444,
  IMG_TYPE_RGB,
  IMG_TYPE_BGR,
  TENSOR_TYPE_S4,
  TENSOR_TYPE_U4,
  TENSOR_TYPE_S8,  // 8
  TENSOR_TYPE_U8,
  TENSOR_TYPE_F16,
  TENSOR_TYPE_S16,
  TENSOR_TYPE_U16,
  TENSOR_TYPE_F32,  // 13
  TENSOR_TYPE_S32,
  TENSOR_TYPE_U32,
  TENSOR_TYPE_F64,
  TENSOR_TYPE_S64,
  TENSOR_TYPE_U64,
  TENSOR_TYPE_MAX
} DataType;

// 浮点转换结果
typedef struct {
  TensorLayout layout;
  int dim[4];
  std::vector<float> value;  // batch * (nhwc), batch for resizer model
} FloatTensor;

class HandLmkOutputParser {
 public:
  HandLmkOutputParser() {}
  ~HandLmkOutputParser() {}

  // 对于roi infer task，每个roi对应一次Parse
  // 因此需要在Parse中实现output和roi的match处理，即当前的Parse对应的是那个roi
  // int32_t Parse(
  //     std::shared_ptr<LandmarksResult> &output,
  //     std::shared_ptr<DNNTensor> &output_tensor,
  //     std::shared_ptr<std::vector<hbDNNRoi>> rois);
  int32_t Parse(
      std::shared_ptr<LandmarksResult> &output,
      std::shared_ptr<DNNTensor> &output_tensor,
      std::shared_ptr<std::vector<hbDNNRoi>> rois);
 private:
  int i_o_stride_ = 4;
  void LmksPostPro(const FloatTensor& float_tensor,
                   const int valid_offset,
                   const int valid_result_idx,
                   const hbDNNRoi& roi,
                   std::shared_ptr<LandmarksResult>& output);                   
  void OutputTensors2FloatTensors(const DNNTensor& tensor,
                                  FloatTensor& float_tensor,
                                  int batch);
};

#endif  // HAND_LMK_OUTPUT_PARSER_H
