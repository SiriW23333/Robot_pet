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

#include "include/gesture_postprocess_util.h"

#include <iostream>

#include "rclcpp/rclcpp.hpp"

namespace inference {

GesturePostProcessUtil* GesturePostProcessUtil::util_instance_ = nullptr;
std::mutex GesturePostProcessUtil::instance_mutex_;

GesturePostProcessUtil* GesturePostProcessUtil::GetInstance() {
  if (util_instance_ == nullptr) {
    std::lock_guard<std::mutex> guard(GesturePostProcessUtil::instance_mutex_);
    if (util_instance_ == nullptr) {
      util_instance_ = new (std::nothrow) GesturePostProcessUtil();
    }
  }
  return util_instance_;
}

int GesturePostProcessUtil::Init(float window_size, int score_size) {
  if (!inited_) {
    cached_scores_map_.clear();
    window_size_ = window_size;
    score_size_ = score_size;
    inited_ = true;
    std::stringstream ss;
    ss << "act post util, window_size: " << window_size_
       << ", score_size: " << score_size_;
    RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());
  } else {
    std::stringstream ss;
    ss << "already inited with window_size: " << window_size_
       << ", score_size_: " << score_size_;
    RCLCPP_INFO(rclcpp::get_logger("post process"), "%s", ss.str().c_str());
  }
  return 0;
}

std::vector<float> GesturePostProcessUtil::GetCachedAvgScore(
    float timestamp, int track_id, std::vector<float> cur_score) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  // current score entry
  CachedScoreEntry cur_entry(timestamp);
  cur_entry.score_.assign(cur_score.begin(), cur_score.end());
  // get data for current track id
  auto iter = cached_scores_map_.find(track_id);
  if (iter == cached_scores_map_.end()) {
    std::deque<CachedScoreEntry> cached_scores;
    cached_scores.push_back(cur_entry);
    cached_scores_map_[track_id] = cached_scores;
    return cur_score;
  } else {
    auto cached_scores = iter->second;
    cached_scores.push_back(cur_entry);
    auto front_timestamp = cached_scores.front().timestamp_;
    while (timestamp - front_timestamp > window_size_) {
      cached_scores.pop_front();
      front_timestamp = cached_scores.front().timestamp_;
    }
    auto avg_score = CalcAvg(cached_scores);
    return avg_score;
  }
}

void GesturePostProcessUtil::Clean(std::vector<int64_t> disappeared_track_ids) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  for (size_t i = 0; i < disappeared_track_ids.size(); ++i) {
    auto disappeared_track_id = disappeared_track_ids.at(i);
    if (cached_scores_map_.find(disappeared_track_id) !=
        cached_scores_map_.end()) {
      cached_scores_map_[disappeared_track_id].clear();
      cached_scores_map_.erase(disappeared_track_id);
    }
  }
}

std::vector<float> GesturePostProcessUtil::CalcAvg(
    std::deque<CachedScoreEntry> data) {
  std::vector<float> avg_score;
  for (int score_idx = 0; score_idx < score_size_; ++score_idx) {
    float sum = 0;
    for (size_t data_idx = 0; data_idx < data.size(); ++data_idx) {
      sum += data[data_idx].score_[score_idx];
    }
    float avg = sum / static_cast<float>(data.size());
    avg_score.push_back(avg);
  }
  return avg_score;
}

}  // namespace inference
