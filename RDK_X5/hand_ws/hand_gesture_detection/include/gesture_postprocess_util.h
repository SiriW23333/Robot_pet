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

#ifndef ACT_POSTPROCESS_UTIL_H_
#define ACT_POSTPROCESS_UTIL_H_

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace inference {

class CachedScoreEntry {
 public:
  explicit CachedScoreEntry(float timestamp) : timestamp_(timestamp) {}
  void Clean() { score_.clear(); }
  std::vector<float> score_;
  float timestamp_;
};

class GesturePostProcessUtil {
 public:
  GesturePostProcessUtil() { inited_ = false; }

  static GesturePostProcessUtil *GetInstance();

  int Init(float window_size, int score_size);

  // timestamp单位秒
  std::vector<float> GetCachedAvgScore(float timestamp,
                                       int track_id,
                                       std::vector<float> cur_score);

  void Clean(std::vector<int64_t> disappeared_track_ids);

 private:
  std::vector<float> CalcAvg(std::deque<CachedScoreEntry> data);

 private:
  std::unordered_map<int, std::deque<CachedScoreEntry>> cached_scores_map_;
  // 单位秒，表示window时间内的数据
  float window_size_;
  int score_size_;
  bool inited_ = false;
  std::mutex map_mutex_;
  static GesturePostProcessUtil *util_instance_;
  static std::mutex instance_mutex_;
};

}  // namespace inference

#endif  // ACT_POSTPROCESS_UTIL_H_
