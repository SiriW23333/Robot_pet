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

#ifndef LMKS_PROCESS_H_
#define LMKS_PROCESS_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "dnn/hb_dnn.h"
#include "include/gesture_data.h"

namespace inference {

struct LmksProcessTensor {
 public:
  LmksProcessTensor(int dim1_, int dim2_, int dim3_, int dim4_)
      : dim1(dim1_), dim2(dim2_), dim3(dim3_), dim4(dim4_) {
    if (!(dim1_ > 0 && dim2_ > 0 && dim3_ > 0 && dim4_ > 0)) {
      std::cout << "Failed to create tensor of shape: " << dim1_ << " " << dim2_
                << " " << dim3_ << " " << dim4_;
    }
    data = std::vector<float>(dim1_ * dim2_ * dim3_ * dim4_, 0);
  }

  float At(int x1, int x2, int x3, int x4) {
    return data[x1 * dim2 * dim3 * dim4 + x2 * dim3 * dim4 + x3 * dim4 + x4];
  }

  void Set(int x1, int x2, int x3, int x4, float value) {
    if (!(x1 >= 0 && x1 < dim1) || !(x2 >= 0 && x2 < dim2) ||
        !(x3 >= 0 && x3 < dim3) || !(x4 >= 0 && x4 < dim4)) {
      std::cout << "Error set tensor at position: " << x1 << " " << x2 << " "
                << x3 << " " << x4 << ", with shape: " << dim1 << " " << dim2
                << " " << dim3 << " " << dim4 << "\n";
    }
    data[x1 * dim2 * dim3 * dim4 + x2 * dim3 * dim4 + x3 * dim4 + x4] = value;
  }

  std::vector<float> data;

 private:
  int dim1;
  int dim2;
  int dim3;
  int dim4;
};

class FeatureSequenceBuffer {
 public:
  FeatureSequenceBuffer() {
    max_len_ = 100;
    len_ = 0;
    feats_ = std::make_shared<inference::LandmarkVector>();
    boxes_ = std::make_shared<inference::BoxVector>();
  }
  explicit FeatureSequenceBuffer(int buff_len_) : len_(0), max_len_(buff_len_) {
    feats_ = std::make_shared<inference::LandmarkVector>();
    boxes_ = std::make_shared<inference::BoxVector>();
  }

  void Update(std::shared_ptr<hbDNNRoi> box,
              std::shared_ptr<inference::Landmarks> kps,
              uint64_t timestamp);

  void GetClipFeatByEnd(std::shared_ptr<inference::LandmarkVector> kpses,
                        std::shared_ptr<inference::BoxVector> boxes,
                        int num,
                        float stride,
                        float margin,
                        uint64_t end);

  void Clean() {
    timestamps_.clear();
    feats_->clear();
    boxes_->clear();
  }

  int GetLatestTs(uint64_t& ts) {
    if (timestamps_.empty()) {
      return -1;
    }
    ts = timestamps_.back();
    return 0;
  }

 private:
  int len_ = 0;
  int max_len_;
  // Float data type has precision loss if the ts value is large number
  std::vector<uint64_t> timestamps_;
  std::shared_ptr<inference::LandmarkVector> feats_;
  std::shared_ptr<inference::BoxVector> boxes_;
};

class LmksProcess {
 public:
  LmksProcess() {}
  ~LmksProcess() {}

  int Init(int num_kps,
           int seq_len,
           float stride,
           float max_gap,
           float kps_norm_scale,
           bool norm_kps_conf,
           int buffer_len = 100) {
    num_kps_ = num_kps;
    seq_len_ = seq_len;
    stride_ = stride;
    max_gap_ = max_gap;
    kps_norm_scale_ = kps_norm_scale;
    buff_len_ = buffer_len;
    norm_kps_conf_ = norm_kps_conf;
    std::stringstream ss;
    ss << "num_kps: " << num_kps << ", seq_len: " << seq_len
       << ", stride: " << stride << ", max_gap: " << max_gap
       << ", buff_len: " << buffer_len
       << ", kps_norm_scale: " << kps_norm_scale_ << "\n";
    RCLCPP_INFO(rclcpp::get_logger("lmk process"), "%s", ss.str().c_str());
    return 0;
  }

  void Update(int track_id,
              std::shared_ptr<hbDNNRoi> box,
              std::shared_ptr<inference::Landmarks> kps,
              uint64_t timestamp);

  void NormKps(std::shared_ptr<inference::LandmarkVector>& kpses,
               const std::shared_ptr<inference::BoxVector>& boxes);

  void GetClipKps(std::shared_ptr<inference::LandmarkVector> kpses,
                  int track_id,
                  uint64_t timestamp);

  void Clean(const std::vector<uint64_t>& disappeared_track_ids);

  void Clean(uint64_t timestamp);

  void Execute(int track_id,
               std::shared_ptr<hbDNNRoi> box,
               std::shared_ptr<inference::Landmarks> kps,
               std::shared_ptr<inference::LandmarkVector> cached_kpses,
               uint64_t timestamp);

 private:
  std::map<int, FeatureSequenceBuffer> track_buffers_;
  int num_kps_;
  int seq_len_;
  float stride_;
  float max_gap_;
  int buff_len_;
  int rotate_degree_;
  bool norm_kps_conf_;
  float max_score_ = -1;
  float kps_norm_scale_ = 1;
  std::mutex map_mutex_;
  // 最近一次更新的ts和当前ts间隔超过10秒，认为track消失，清理缓存
  uint64_t clean_ts_diff_ms_ = 33 * 30 * 10;
};

}  // namespace inference

#endif  // LMKS_PROCESS_H_
