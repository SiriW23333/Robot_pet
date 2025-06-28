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

#ifndef GESTURE_POSTPROCESS_H_
#define GESTURE_POSTPROCESS_H_

#include <memory>
#include <string>
#include <vector>

#include "include/gesture_data.h"
#include "include/gesture_postprocess_util.h"

namespace inference {

class GesturePostProcess {
 public:
  explicit GesturePostProcess(std::string json_str, float threshold) {
    threshold_ = threshold;
    Init(json_str);
  }

  ~GesturePostProcess() {}

  int Init(const std::string &json_str);

  std::shared_ptr<GestureRes> Execute(
      const std::shared_ptr<DNNTensor> &output_tensor,
      const uint64_t &track_id,
      const uint64_t &timestamp);

 private:
  void InitThresholds(std::string &thresholds_str);

  void InitMergeGroups(std::string &merge_groups_str);

  GestureRes ActPostPro(std::vector<FloatTensor> &float_tensors,
                        uint64_t timestamp,
                        uint64_t track_id);

  void OutputTensors2FloatTensors(const DNNTensor &tensor,
                                  FloatTensor &float_tensor,
                                  int batch);

 private:
  int en_score_avg_ = 0;
  float threshold_ = 0.5;  // 0.95;
  std::vector<float> thresholds_ = {0.95};
  std::vector<std::vector<int>> merge_groups_ = {
      {0, 1, 2, 3, 6, 12, 13, 14, 15, 16},
      {},
      {4},
      {5},
      {7},
      {8, 9, 10},
      {},
      {},
      {},
      {},
      {},
      {17},
      {18},
      {19},
      {20},
      {11},
      {21},
      {22}};

  float window_size_ = 0.5;
};

}  // namespace inference

#endif  // GESTURE_POSTPROCESS_H_
