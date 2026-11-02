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
#include "hydra/utils/observation_recorder.h"

#include <config_utilities/config.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "hydra/input/camera.h"
#include "hydra/input/input_data.h"
#include "hydra/input/input_packet.h"

namespace hydra {

namespace {

std::string escapeJson(const std::string& str) {
  std::stringstream ss;
  for (const char c : str) {
    switch (c) {
      case '"':
        ss << "\\\"";
        break;
      case '\\':
        ss << "\\\\";
        break;
      case '\n':
        ss << "\\n";
        break;
      case '\r':
        ss << "\\r";
        break;
      case '\t':
        ss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          ss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(c) << std::dec;
        } else {
          ss << c;
        }
    }
  }
  return ss.str();
}

std::string quoted(const std::string& str) { return "\"" + escapeJson(str) + "\""; }

std::string number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }

  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return ss.str();
}

//! Pose as {"t": [x, y, z], "q_wxyz": [w, x, y, z]}
std::string poseJson(const Eigen::Isometry3d& pose) {
  const Eigen::Vector3d t = pose.translation();
  Eigen::Quaterniond q(pose.linear());
  q.normalize();
  std::stringstream ss;
  ss << "{\"t\": [" << number(t.x()) << ", " << number(t.y()) << ", "
     << number(t.z()) << "], \"q_wxyz\": [" << number(q.w()) << ", " << number(q.x())
     << ", " << number(q.y()) << ", " << number(q.z()) << "]}";
  return ss.str();
}

std::string optionalPath(const std::optional<std::string>& path) {
  return path ? quoted(*path) : "null";
}

std::string frameName(uint64_t frame_id) {
  std::stringstream ss;
  ss << std::setw(8) << std::setfill('0') << frame_id;
  return ss.str();
}

}  // namespace

void declare_config(ObservationRecorder::Config& config) {
  using namespace config;
  name("ObservationRecorder::Config");
  field(config.enable, "enable");
  field<Path>(config.output_path, "output_path");
  field(config.max_queue_frames, "max_queue_frames");
  field(config.max_queue_bytes, "max_queue_bytes");
  field(config.queue_full_policy, "queue_full_policy");
  field(config.save_depth, "save_depth");
  field(config.depth_format, "depth_format");
  field(config.png_compression, "png_compression");
  field(config.num_writer_threads, "num_writer_threads");
  if (config.enable) {
    checkCondition(!config.output_path.empty(), "output_path required when enabled");
    checkCondition(config.max_queue_frames > 0, "max_queue_frames must be positive");
  }
  checkCondition(
      config.queue_full_policy == "block" || config.queue_full_policy == "drop",
      "queue_full_policy must be 'block' or 'drop'");
  checkCondition(
      config.depth_format == "tiff_f32" || config.depth_format == "png_u16_mm",
      "depth_format must be 'tiff_f32' or 'png_u16_mm'");
  checkInRange(config.png_compression, 0, 9, "png_compression");
  checkCondition(config.num_writer_threads > 0, "num_writer_threads must be positive");
}

ObservationRecorder::ObservationRecorder(const Config& config)
    : config(config::checkValid(config)),
      block_on_full_(config.queue_full_policy == "block") {
  if (!config.enable) {
    stopped_ = true;
    return;
  }

  std::error_code ec;
  std::filesystem::create_directories(config.output_path / "rgb", ec);
  if (!ec && config.save_depth) {
    std::filesystem::create_directories(config.output_path / "depth", ec);
  }

  if (ec) {
    LOG(ERROR) << "[Observation Recorder] Failed to create output directories in "
               << config.output_path << ": " << ec.message();
  }

  const auto frames_path = config.output_path / "frames_raw.jsonl";
  if (std::filesystem::exists(frames_path)) {
    LOG(WARNING) << "[Observation Recorder] Overwriting existing " << frames_path;
  }

  frames_file_.open(frames_path, std::ios::out | std::ios::trunc);
  if (!frames_file_.good()) {
    LOG(ERROR) << "[Observation Recorder] Unable to open " << frames_path
               << "; frame metadata will not be saved!";
  }

  for (size_t i = 0; i < config.num_writer_threads; ++i) {
    write_threads_.emplace_back(&ObservationRecorder::writeSpin, this);
  }
  LOG(INFO) << "[Observation Recorder] Recording observations to "
            << config.output_path << " (queue_full_policy=" << config.queue_full_policy
            << ")";
}

ObservationRecorder::~ObservationRecorder() { stop(); }

std::string ObservationRecorder::rgbPath(uint64_t frame_id) {
  return "rgb/" + frameName(frame_id) + ".png";
}

std::string ObservationRecorder::depthPath(uint64_t frame_id) const {
  const std::string ext = config.depth_format == "tiff_f32" ? ".tiff" : ".png";
  return "depth/" + frameName(frame_id) + ext;
}

void ObservationRecorder::recordInvalid(const InputPacket& packet,
                                        const std::string& reason) {
  if (!config.enable) {
    return;
  }

  PendingFrame frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_shutdown_) {
      ++stats_.packets_after_stop;
      return;
    }

    ++stats_.packets_seen;
    ++stats_.packets_invalid;
    frame.info.frame_id = next_frame_id_++;
  }

  frame.info.timestamp_ns = packet.timestamp_ns;
  frame.info.world_T_body = packet.world_T_body();
  if (packet.sensor_input) {
    frame.info.sensor_name = packet.sensor_input->sensor_name;
    frame.info.sensor_frame = packet.sensor_input->sensor_frame;
  }

  frame.status = "invalid_input";
  frame.error = reason;
  enqueueMetadata(std::move(frame));
}

bool ObservationRecorder::record(const InputPacket& packet,
                                 const InputData& data,
                                 bool triggered_full_update) {
  if (!config.enable) {
    return false;
  }

  const auto& sensor = data.getSensor();
  const bool has_color = !data.color_image.empty() && data.color_image.type() == CV_8UC3;
  const bool has_depth = !data.depth_image.empty();

  PendingFrame frame;
  frame.info.sensor_name = sensor.name;
  frame.info.sensor_frame = packet.sensor_input ? packet.sensor_input->sensor_frame : "";
  frame.info.timestamp_ns = data.timestamp_ns;
  frame.info.world_T_body = data.world_T_body;
  frame.info.body_T_sensor = sensor.body_T_sensor();
  frame.info.triggered_full_update = triggered_full_update;
  frame.info.has_depth = has_depth;
  if (has_color) {
    frame.info.width = data.color_image.cols;
    frame.info.height = data.color_image.rows;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_shutdown_) {
      ++stats_.packets_after_stop;
      return false;
    }

    ++stats_.packets_seen;
    if (!calibrations_.count(sensor.name)) {
      calibrations_[sensor.name] = calibrationJson(sensor);
    }

    if (!has_color) {
      ++stats_.packets_without_color;
      return false;
    }

    ++stats_.frames_received;
    frame.info.frame_id = next_frame_id_++;
    frame.info.sensor_sequence = sensor_sequences_[sensor.name]++;
  }

  // deep copies so that the writer never shares buffers with the pipeline
  frame.rgb = data.color_image.clone();
  if (has_depth && config.save_depth) {
    frame.depth = data.depth_image.clone();
  }

  frame.num_bytes = frame.rgb.total() * frame.rgb.elemSize() +
                    frame.depth.total() * frame.depth.elemSize();

  std::unique_lock<std::mutex> lock(mutex_);
  const auto has_room = [this, &frame]() {
    if (should_shutdown_) {
      return true;
    }

    if (queued_image_frames_ == 0) {
      return true;  // always accept a single frame, even when larger than the limit
    }

    const bool frames_ok = queued_image_frames_ < config.max_queue_frames;
    const bool bytes_ok = !config.max_queue_bytes ||
                          queued_bytes_ + frame.num_bytes <= config.max_queue_bytes;
    return frames_ok && bytes_ok;
  };

  if (!has_room()) {
    if (!block_on_full_) {
      ++stats_.frames_dropped_queue_full;
      LOG_EVERY_N(WARNING, 20) << "[Observation Recorder] Queue full, dropped frame "
                               << frame.info.frame_id << " @ "
                               << frame.info.timestamp_ns << " [ns] (total dropped: "
                               << stats_.frames_dropped_queue_full << ")";
      frame.rgb.release();
      frame.depth.release();
      frame.num_bytes = 0;
      frame.status = "dropped_queue_full";
      queue_.push_back(std::move(frame));
      lock.unlock();
      queue_cv_.notify_all();
      return false;
    }

    const auto start = std::chrono::steady_clock::now();
    queue_cv_.wait(lock, has_room);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    stats_.total_block_time_s += elapsed.count();
  }

  if (should_shutdown_) {
    // stop() was called while waiting for space: the writer may already be gone
    ++stats_.frames_rejected_on_shutdown;
    LOG(ERROR) << "[Observation Recorder] Frame " << frame.info.frame_id
               << " arrived during shutdown and was not saved";
    return false;
  }

  ++stats_.frames_enqueued;
  ++queued_image_frames_;
  queued_bytes_ += frame.num_bytes;
  stats_.max_queue_frames_seen = std::max(stats_.max_queue_frames_seen, queued_image_frames_);
  stats_.max_queue_bytes_seen = std::max(stats_.max_queue_bytes_seen, queued_bytes_);
  queue_.push_back(std::move(frame));
  lock.unlock();
  queue_cv_.notify_all();
  return true;
}

void ObservationRecorder::enqueueMetadata(PendingFrame&& frame) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(frame));
  }
  queue_cv_.notify_all();
}

void ObservationRecorder::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ && write_threads_.empty()) {
      return;
    }

    should_shutdown_ = true;
  }

  queue_cv_.notify_all();
  for (auto& thread : write_threads_) {
    thread.join();  // the write threads drain the queue before exiting
  }
  write_threads_.clear();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
  }

  frames_file_.flush();
  frames_file_.close();
  writeCalibration();
  writeStats();

  const auto s = stats();
  LOG(INFO) << "[Observation Recorder] Stopped: received=" << s.frames_received
            << ", written=" << s.frames_written
            << ", write_failed=" << s.frames_write_failed
            << ", dropped=" << s.frames_dropped_queue_full
            << ", invalid=" << s.packets_invalid
            << ", without_color=" << s.packets_without_color;
  LOG_IF(ERROR, s.frames_write_failed || s.metadata_write_failed)
      << "[Observation Recorder] " << s.frames_write_failed << " frame(s) and "
      << s.metadata_write_failed << " metadata line(s) failed to write!";
}

ObservationRecorder::Stats ObservationRecorder::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void ObservationRecorder::writeSpin() {
  while (true) {
    PendingFrame frame;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_cv_.wait(lock, [this] { return should_shutdown_ || !queue_.empty(); });
      if (queue_.empty()) {
        break;  // only exit once shutdown is requested and every frame is written
      }

      frame = std::move(queue_.front());
      queue_.pop_front();
      frame.write_seq = next_pop_seq_++;
    }

    writeFrame(frame);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (frame.status.empty()) {
        --queued_image_frames_;
        queued_bytes_ -= frame.num_bytes;
      }
    }

    queue_cv_.notify_all();
  }
}

bool ObservationRecorder::writeImage(const std::filesystem::path& path,
                                     const cv::Mat& image,
                                     std::string& error) const {
  // write to a temporary file first so a partially written image never appears
  // under its final name
  const auto tmp_path = path.parent_path() / (path.stem().string() + ".tmp" +
                                              path.extension().string());
  try {
    std::vector<int> params;
    if (path.extension() == ".png") {
      params = {cv::IMWRITE_PNG_COMPRESSION, config.png_compression};
    }

    if (!cv::imwrite(tmp_path.string(), image, params)) {
      error = "cv::imwrite failed for " + path.string();
      return false;
    }
  } catch (const cv::Exception& e) {
    error = "cv::imwrite threw for " + path.string() + ": " + e.what();
    return false;
  }

  std::error_code ec;
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    error = "rename failed for " + path.string() + ": " + ec.message();
    return false;
  }

  return true;
}

void ObservationRecorder::writeFrame(PendingFrame& frame) {
  if (!frame.status.empty()) {
    writeLine(toJsonLine(frame.info, frame.status, std::nullopt, std::nullopt, frame.error),
              frame.write_seq);
    return;
  }

  std::string error;
  const auto rgb_rel = rgbPath(frame.info.frame_id);
  cv::Mat bgr;
  cv::cvtColor(frame.rgb, bgr, cv::COLOR_RGB2BGR);  // imwrite expects BGR order
  bool rgb_ok = writeImage(config.output_path / rgb_rel, bgr, error);

  std::optional<std::string> depth_rel;
  bool depth_ok = true;
  if (!frame.depth.empty()) {
    depth_rel = depthPath(frame.info.frame_id);
    cv::Mat depth_out;
    if (config.depth_format == "tiff_f32") {
      depth_out = frame.depth;
    } else {
      // invalid and out-of-range depth is stored as 0
      cv::Mat valid = (frame.depth == frame.depth) & (frame.depth > 0.0f) &
                      (frame.depth < 65.535f);
      cv::Mat mm;
      frame.depth.convertTo(mm, CV_16UC1, 1000.0);
      depth_out = cv::Mat::zeros(frame.depth.size(), CV_16UC1);
      mm.copyTo(depth_out, valid);
    }

    std::string depth_error;
    depth_ok = writeImage(config.output_path / *depth_rel, depth_out, depth_error);
    if (!depth_ok) {
      error += (error.empty() ? "" : "; ") + depth_error;
    }
  }

  const bool ok = rgb_ok && depth_ok;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ok) {
      ++stats_.frames_written;
    } else {
      ++stats_.frames_write_failed;
    }

    if (depth_rel && depth_ok) {
      ++stats_.depth_written;
    }
  }

  if (!ok) {
    LOG(ERROR) << "[Observation Recorder] Failed to write frame " << frame.info.frame_id
               << ": " << error;
  }

  writeLine(toJsonLine(frame.info,
                       ok ? "written" : "write_failed",
                       rgb_ok ? std::optional<std::string>(rgb_rel) : std::nullopt,
                       depth_ok ? depth_rel : std::nullopt,
                       error),
            frame.write_seq);
}

void ObservationRecorder::writeLine(const std::string& line, uint64_t write_seq) {
  std::unique_lock<std::mutex> line_lock(line_mutex_);
  line_cv_.wait(line_lock, [&] { return next_line_seq_ == write_seq; });
  frames_file_ << line << "\n";
  frames_file_.flush();
  ++next_line_seq_;
  line_cv_.notify_all();
  if (!frames_file_.good()) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.metadata_write_failed;
    LOG_EVERY_N(ERROR, 20) << "[Observation Recorder] Failed to write frame metadata";
  }
}

std::string ObservationRecorder::toJsonLine(const FrameInfo& info,
                                            const std::string& status,
                                            const std::optional<std::string>& rgb_path,
                                            const std::optional<std::string>& depth_path,
                                            const std::string& error) {
  const Eigen::Isometry3d world_T_sensor = info.world_T_body * info.body_T_sensor;
  std::stringstream ss;
  ss << "{\"frame_id\": " << info.frame_id << ", \"status\": " << quoted(status)
     << ", \"sensor\": " << quoted(info.sensor_name)
     << ", \"sensor_frame\": " << quoted(info.sensor_frame)
     << ", \"sensor_sequence\": " << info.sensor_sequence
     << ", \"timestamp_ns\": " << info.timestamp_ns
     << ", \"timestamp_ns_str\": " << quoted(std::to_string(info.timestamp_ns))
     << ", \"rgb_path\": " << optionalPath(rgb_path)
     << ", \"depth_path\": " << optionalPath(depth_path) << ", \"width\": " << info.width
     << ", \"height\": " << info.height
     << ", \"has_depth\": " << (info.has_depth ? "true" : "false")
     << ", \"triggered_full_update\": " << (info.triggered_full_update ? "true" : "false")
     << ", \"world_T_body\": " << poseJson(info.world_T_body)
     << ", \"body_T_sensor\": " << poseJson(info.body_T_sensor)
     << ", \"world_T_sensor\": " << poseJson(world_T_sensor)
     << ", \"error\": " << (error.empty() ? "null" : quoted(error)) << "}";
  return ss.str();
}

std::string ObservationRecorder::calibrationJson(const Sensor& sensor) {
  std::stringstream ss;
  ss << "{\"name\": " << quoted(sensor.name);
  const auto camera = dynamic_cast<const Camera*>(&sensor);
  if (camera) {
    const auto& c = camera->getConfig();
    ss << ", \"model\": \"pinhole\", \"width\": " << c.width << ", \"height\": " << c.height
       << ", \"fx\": " << number(c.fx) << ", \"fy\": " << number(c.fy)
       << ", \"cx\": " << number(c.cx) << ", \"cy\": " << number(c.cy)
       << ", \"distortion_model\": \"none\""
       << ", \"distortion_note\": \"Hydra Camera has no distortion parameters; inputs "
          "are assumed to be rectified\"";
  } else {
    ss << ", \"model\": \"non_camera\"";
  }

  ss << ", \"min_range\": " << number(sensor.min_range())
     << ", \"max_range\": " << number(sensor.max_range())
     << ", \"body_T_sensor\": " << poseJson(sensor.body_T_sensor()) << "}";
  return ss.str();
}

void ObservationRecorder::writeCalibration() const {
  if (!config.enable) {
    return;
  }

  std::map<std::string, std::string> calibrations;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    calibrations = calibrations_;
  }

  const auto path = config.output_path / "camera_calib.json";
  std::ofstream fout(path);
  fout << "{\"convention\": \"world_T_sensor = world_T_body * body_T_sensor; "
          "camera frame is x right, y down, z forward\", \"sensors\": {";
  bool first = true;
  for (const auto& [name, json] : calibrations) {
    fout << (first ? "" : ", ") << quoted(name) << ": " << json;
    first = false;
  }
  fout << "}}\n";
  LOG_IF(ERROR, !fout.good()) << "[Observation Recorder] Failed to write " << path;
}

void ObservationRecorder::writeStats() const {
  if (!config.enable) {
    return;
  }

  const auto s = stats();
  const auto path = config.output_path / "recorder_stats.json";
  std::ofstream fout(path);
  fout << "{\"scope\": \"inputs popped by the reconstruction module while recording was "
          "enabled; frames dropped upstream (sensor driver, ROS sync, "
          "input_separation_s, missing body pose) are not counted\""
       << ", \"queue_full_policy\": " << quoted(config.queue_full_policy)
       << ", \"packets_seen\": " << s.packets_seen
       << ", \"packets_invalid\": " << s.packets_invalid
       << ", \"packets_without_color\": " << s.packets_without_color
       << ", \"frames_received\": " << s.frames_received
       << ", \"frames_enqueued\": " << s.frames_enqueued
       << ", \"frames_dropped_queue_full\": " << s.frames_dropped_queue_full
       << ", \"frames_written\": " << s.frames_written
       << ", \"frames_rejected_on_shutdown\": " << s.frames_rejected_on_shutdown
       << ", \"packets_after_stop\": " << s.packets_after_stop
       << ", \"frames_write_failed\": " << s.frames_write_failed
       << ", \"depth_written\": " << s.depth_written
       << ", \"metadata_write_failed\": " << s.metadata_write_failed
       << ", \"max_queue_frames_seen\": " << s.max_queue_frames_seen
       << ", \"max_queue_bytes_seen\": " << s.max_queue_bytes_seen
       << ", \"total_block_time_s\": " << number(s.total_block_time_s) << "}\n";
  LOG_IF(ERROR, !fout.good()) << "[Observation Recorder] Failed to write " << path;
}

}  // namespace hydra
