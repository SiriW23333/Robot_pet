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

#include "include/lmks_process.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "include/gesture_data.h"
#include "rclcpp/rclcpp.hpp"

namespace inference {

void FeatureSequenceBuffer::Update(std::shared_ptr<hbDNNRoi> box,
                                   std::shared_ptr<inference::Landmarks> kps,
                                   uint64_t timestamp) {
  while (len_ > max_len_) {
    RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"),
                 "overflow, removing..., len: %d, max_len_: %d",
                 len_,
                 max_len_);
    timestamps_.erase(timestamps_.begin());
    feats_->erase(feats_->begin());
    boxes_->erase(boxes_->begin());
    len_ -= 1;
  }
  feats_->push_back(kps);
  boxes_->push_back(box);
  timestamps_.push_back(timestamp);
  len_ += 1;
}

void FeatureSequenceBuffer::GetClipFeatByEnd(
    std::shared_ptr<inference::LandmarkVector> kpses,
    std::shared_ptr<inference::BoxVector> boxes,
    int num,
    float stride,
    float margin,
    uint64_t end) {
  if (!kpses || !boxes || num <= 0) {
    RCLCPP_ERROR(rclcpp::get_logger("lmk pro"), "Invalid inputs! num: %d", num);
    return;
  }
  kpses->clear();
  boxes->clear();
  if (len_ < 1) {
    RCLCPP_INFO(rclcpp::get_logger("lmk pro"), "len_ < 1, return...");
    return;
  }

  // std::stringstream ss;
  // ss << "num: " << num << " ";
  // ss << "stride: " << stride << " ";
  // ss << "end: " << end << " ";
  // ss << "len_: " << len_ << " ";
  // ss << "max_len_: " << max_len_ << " ";
  // ss << "\n";
  std::vector<int> clip_idxs(num, -1);
  for (int frame_idx = 0; frame_idx < num; ++frame_idx) {
    // timestamp for expected candidate
    uint64_t curtime =
        end - static_cast<uint64_t>(static_cast<float>(frame_idx) * stride);
    // ss << "frame_idx: " << frame_idx << ", curtime: " << curtime << "\n";
    // time gap
    uint64_t restime = 1e10;
    uint64_t time_diff = 0;
    for (int idx = len_ - 1; idx >= 0; --idx) {
      if (timestamps_[idx] > curtime) {
        time_diff = timestamps_[idx] - curtime;
      } else {
        time_diff = curtime - timestamps_[idx];
      }
      // ss << "\t timestamps_[" << idx << "]: " << timestamps_[idx] << ",
      // time_diff: " << time_diff
      // << ", restime: " << restime << ", margin: " << margin << " ";
      if (time_diff < restime) {
        restime = time_diff;
        // ss << ", update restime: " << restime << " ";
        if (restime < margin) {
          clip_idxs[num - 1 - frame_idx] = idx;
          // ss << ", update clip_idxs idx[" << num - 1 - frame_idx << "]: " <<
          // clip_idxs[num - 1 - frame_idx];
        }
      } else {
        // ss << ", update done";
        break;
      }
      // ss << "\n";
    }
    // ss << "\n";
  }

  // RCLCPP_INFO(rclcpp::get_logger("lmk process"), "%s", ss.str().c_str());

  int maxValue = clip_idxs[0];
  int minValue = clip_idxs[0];
  for (auto& item : clip_idxs) {
    if (item < minValue) {
      minValue = item;
    }
    if (item > maxValue) {
      maxValue = item;
    }
  }
  if (maxValue >= len_ || minValue < 0) {
    std::stringstream ss;
    ss << "Failed to get clip kps by end, invalid index. "
       << "max_idx: " << maxValue << " min_idx: " << minValue
       << " length: " << len_;
    RCLCPP_DEBUG(rclcpp::get_logger("lmk process"), "%s", ss.str().c_str());
    return;
  }

  for (auto& idx : clip_idxs) {
    // 深拷贝
    kpses->push_back(std::make_shared<Landmarks>(*feats_->at(idx)));
    boxes->push_back(std::make_shared<hbDNNRoi>(*boxes_->at(idx)));
  }
}

void LmksProcess::Update(int track_id,
                         std::shared_ptr<hbDNNRoi> box,
                         std::shared_ptr<inference::Landmarks> kps,
                         uint64_t timestamp) {
  std::stringstream ss;
  ss << "Update track_id: " << track_id << ", timestamp: " << timestamp;
  RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"), "%s", ss.str().c_str());

  if (track_buffers_.find(track_id) == track_buffers_.end()) {
    FeatureSequenceBuffer buffer(buff_len_);
    track_buffers_[track_id] = buffer;
  }
  track_buffers_[track_id].Update(box, kps, timestamp);
}

void LmksProcess::NormKps(std::shared_ptr<inference::LandmarkVector>& kpses,
                          const std::shared_ptr<inference::BoxVector>& boxes) {
  if (!kpses || !boxes) {
    RCLCPP_ERROR(rclcpp::get_logger("lmk pro"), "NormKps invalid inputs!");
    return;
  }

  if (kpses->size() != static_cast<uint32_t>(seq_len_)) {
    RCLCPP_ERROR(rclcpp::get_logger("lmk pro"),
                 "kps len is unmatch: %ld, %d",
                 kpses->size(),
                 seq_len_);
    return;
  }

  RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"),
               "kpses->size: %ld, boxes->size: %ld",
               kpses->size(),
               boxes->size());

  if (!kpses->front() || !boxes->front()) {
    RCLCPP_ERROR(rclcpp::get_logger("lmk pro"), "Invalid inputs");
    return;
  }

  auto first_kps = kpses->at(0);
  inference::HobotPoint center = first_kps->at(9);
  max_score_ = -1;

  for (std::shared_ptr<inference::Landmarks>& kps : *kpses) {
    for (int i = 0; i < num_kps_; ++i) {
      kps->at(i).x -= center.x;
      kps->at(i).y -= center.y;
      if (kps->at(i).score > max_score_) {
        max_score_ = kps->at(i).score;
      }
    }
  }

  if (max_score_ >= 2.0 || norm_kps_conf_) {
    for (std::shared_ptr<inference::Landmarks>& kps : *kpses) {
      for (int i = 0; i < num_kps_; ++i) {
        kps->at(i).score /= kps_norm_scale_;
      }
    }
  }
  float max_width = -1;
  float max_height = -1;
  for (auto p_box : *boxes) {
    const auto& box = p_box;
    float cur_width = box->right - box->left;
    float cur_height = box->bottom - box->top;
    if (cur_width > max_width) {
      max_width = cur_width;
    }
    if (cur_height > max_height) {
      max_height = cur_height;
    }
  }
  float max_border = std::max(max_width, max_height);

  for (std::shared_ptr<Landmarks>& p_kps : *kpses) {
    std::shared_ptr<Landmarks>& kps = p_kps;
    for (int i = 0; i < num_kps_; ++i) {
      kps->at(i).x /= max_border;
      kps->at(i).y /= max_border;
    }
  }
}

void LmksProcess::GetClipKps(std::shared_ptr<inference::LandmarkVector> kpses,
                             int track_id,
                             uint64_t times_tamp) {
  auto boxes = std::make_shared<inference::BoxVector>();
  track_buffers_[track_id].GetClipFeatByEnd(
      kpses, boxes, seq_len_, stride_, max_gap_, times_tamp);
  if (kpses->size() < 1) {
    RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"), "No clip kps get");
    return;
  }

  RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"), "Got clip feat by end");
  NormKps(kpses, boxes);
}

void LmksProcess::Clean(const std::vector<uint64_t>& disappeared_track_ids) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"),
               "data preprocessor with disappeared_track_ids clean() called");
  for (const auto& disappeared_track_id : disappeared_track_ids) {
    if (track_buffers_.find(disappeared_track_id) != track_buffers_.end()) {
      track_buffers_[disappeared_track_id].Clean();
      track_buffers_.erase(disappeared_track_id);
    }
  }
}

void LmksProcess::Clean(uint64_t timestamp) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  RCLCPP_DEBUG(rclcpp::get_logger("lmk pro"),
               "data preprocessor with timestamp clean() called");
  for (std::map<int, FeatureSequenceBuffer>::iterator itr =
           track_buffers_.begin();
       itr != track_buffers_.end();
       itr++) {
    uint64_t track_last_update_ts = 0;
    if (itr->second.GetLatestTs(track_last_update_ts) < 0) {
      continue;
    }
    if (timestamp - track_last_update_ts >= clean_ts_diff_ms_) {
      int disappeared_track_id = itr->first;
      itr++;
      track_buffers_[disappeared_track_id].Clean();
      track_buffers_.erase(disappeared_track_id);
    }
    if (itr == track_buffers_.end()) {
      break;
    }
  }
}

void LmksProcess::Execute(
    int track_id,
    std::shared_ptr<hbDNNRoi> box,
    std::shared_ptr<inference::Landmarks> kps,
    std::shared_ptr<inference::LandmarkVector> cached_kpses,
    uint64_t timestamp) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  Update(track_id, box, kps, timestamp);
  GetClipKps(cached_kpses, track_id, timestamp);
}

}  // namespace inference
