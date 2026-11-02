/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <Eigen/Geometry>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace hydra {

struct InputData;
struct InputPacket;
class Sensor;

/**
 * @brief Records every valid color observation that reaches the reconstruction
 * module (with depth and capture pose) to disk on a background thread.
 *
 * Scope: frames are recorded after they were popped from the active window input
 * queue and successfully converted to InputData. Frames dropped before this point
 * (ROS synchronization, receiver input_separation_s, missing body pose in the input
 * module) are never seen and are NOT recorded.
 *
 * Output layout (all under output_path):
 *  - rgb/<frame_id>.png            8-bit color image (stored as standard PNG RGB)
 *  - depth/<frame_id>.tiff|png     depth image (32-bit float meters or 16-bit mm)
 *  - frames_raw.jsonl              one line per recorded/dropped/failed frame
 *  - camera_calib.json             intrinsics and extrinsics per sensor
 *  - recorder_stats.json           counters, written on stop()
 */
class ObservationRecorder {
 public:
  struct Config {
    //! Turn recording on or off
    bool enable = false;
    //! Directory to write observations to (required when enabled)
    std::filesystem::path output_path;
    //! Maximum number of frames with image data waiting to be written
    size_t max_queue_frames = 32;
    //! Maximum number of image bytes waiting to be written (0 to disable)
    size_t max_queue_bytes = 512 * 1024 * 1024;
    //! Behavior when the queue is full: "block" (keep every frame, may stall the
    //! reconstruction thread) or "drop" (never stall, frame is reported as dropped)
    std::string queue_full_policy = "block";
    //! Save depth images when present
    bool save_depth = true;
    //! Depth encoding: "tiff_f32" (lossless meters) or "png_u16_mm" (millimeters)
    std::string depth_format = "tiff_f32";
    //! PNG compression level [0, 9]
    int png_compression = 1;
    //! Threads encoding and writing queued frames in parallel; frame metadata lines are
    //! still written in queue order
    size_t num_writer_threads = 1;
  } const config;

  struct Stats {
    //! Packets popped by the reconstruction module while recording was enabled
    size_t packets_seen = 0;
    //! Packets that could not be converted into valid input data
    size_t packets_invalid = 0;
    //! Valid packets without an 8-bit three-channel color image
    size_t packets_without_color = 0;
    //! Valid color frames handed to the recorder
    size_t frames_received = 0;
    //! Frames accepted into the write queue
    size_t frames_enqueued = 0;
    //! Frames rejected because the queue was full (policy "drop")
    size_t frames_dropped_queue_full = 0;
    //! Frames that were waiting for queue space when stop() was called (not saved)
    size_t frames_rejected_on_shutdown = 0;
    //! Packets passed to the recorder after stop() (not counted elsewhere)
    size_t packets_after_stop = 0;
    //! Frames whose rgb (and depth if requested) were written successfully
    size_t frames_written = 0;
    //! Frames where at least one file failed to write
    size_t frames_write_failed = 0;
    //! Depth images written
    size_t depth_written = 0;
    //! Metadata lines that failed to write
    size_t metadata_write_failed = 0;
    //! Largest number of frames waiting in the queue
    size_t max_queue_frames_seen = 0;
    //! Largest number of bytes waiting in the queue
    size_t max_queue_bytes_seen = 0;
    //! Total time callers spent waiting for queue space [s]
    double total_block_time_s = 0.0;
  };

  //! Metadata captured for each frame (images are kept separately)
  struct FrameInfo {
    uint64_t frame_id = 0;
    uint64_t sensor_sequence = 0;
    std::string sensor_name;
    std::string sensor_frame;
    uint64_t timestamp_ns = 0;
    Eigen::Isometry3d world_T_body = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d body_T_sensor = Eigen::Isometry3d::Identity();
    bool triggered_full_update = false;
    int width = 0;
    int height = 0;
    bool has_depth = false;
  };

  explicit ObservationRecorder(const Config& config);

  ~ObservationRecorder();

  /**
   * @brief Report a packet that could not be converted to input data
   */
  void recordInvalid(const InputPacket& packet, const std::string& reason);

  /**
   * @brief Record a converted input (images are deep-copied before returning)
   * @param packet Original input packet (for the sensor frame)
   * @param data Converted input data (color is RGB, depth is float meters)
   * @param triggered_full_update Whether this input produced an active window output
   * @returns Whether or not the frame was accepted for writing
   */
  bool record(const InputPacket& packet,
              const InputData& data,
              bool triggered_full_update);

  /**
   * @brief Write all pending frames, the calibration file and stats, and stop
   */
  void stop();

  Stats stats() const;

  const std::filesystem::path& outputPath() const { return config.output_path; }

  //! Relative path (from output_path) of the rgb image for a frame
  static std::string rgbPath(uint64_t frame_id);

  //! Relative path (from output_path) of the depth image for a frame
  std::string depthPath(uint64_t frame_id) const;

  //! Serialize the calibration of a sensor as a json object
  static std::string calibrationJson(const Sensor& sensor);

  //! Serialize the frame metadata as a single json line
  static std::string toJsonLine(const FrameInfo& info,
                                const std::string& status,
                                const std::optional<std::string>& rgb_path,
                                const std::optional<std::string>& depth_path,
                                const std::string& error = "");

 protected:
  struct PendingFrame {
    FrameInfo info;
    cv::Mat rgb;
    cv::Mat depth;
    size_t num_bytes = 0;
    //! Metadata-only entries (dropped / invalid) skip image writing
    std::string status;
    std::string error;
    //! Position in the write order (assigned when a writer takes the frame)
    uint64_t write_seq = 0;
  };

  void writeSpin();

  void writeFrame(PendingFrame& frame);

  void writeLine(const std::string& line, uint64_t write_seq);

  void writeStats() const;

  void writeCalibration() const;

  bool writeImage(const std::filesystem::path& path,
                  const cv::Mat& image,
                  std::string& error) const;

  void enqueueMetadata(PendingFrame&& frame);

  mutable std::mutex mutex_;
  std::condition_variable queue_cv_;
  std::deque<PendingFrame> queue_;
  size_t queued_image_frames_ = 0;
  size_t queued_bytes_ = 0;
  bool should_shutdown_ = false;
  bool stopped_ = false;
  bool block_on_full_ = true;

  Stats stats_;
  uint64_t next_frame_id_ = 0;
  std::map<std::string, uint64_t> sensor_sequences_;
  //! Serialized calibration per sensor, captured when the sensor is first seen
  std::map<std::string, std::string> calibrations_;

  std::ofstream frames_file_;
  std::vector<std::thread> write_threads_;
  //! Metadata lines are emitted in queue order even when frames finish out of order
  uint64_t next_pop_seq_ = 0;
  uint64_t next_line_seq_ = 0;
  std::mutex line_mutex_;
  std::condition_variable line_cv_;
};

void declare_config(ObservationRecorder::Config& config);

}  // namespace hydra
