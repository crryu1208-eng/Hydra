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
#include <gtest/gtest.h>

#include <cstdio>
#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>
#include <hydra/input/input_packet.h>
#include <hydra/input/sensor_input_packet.h>
#include <hydra/utils/observation_recorder.h>

#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

namespace hydra {

namespace {

std::shared_ptr<Camera> makeCamera(const std::string& name) {
  Camera::Config config;
  config.width = 4;
  config.height = 3;
  config.cx = 2.0f;
  config.cy = 1.5f;
  config.fx = 3.0f;
  config.fy = 3.0f;
  config.min_range = 0.1;  // Sensor config requires min_range > 0
  config.max_range = 10.0;
  ParamSensorExtrinsics::Config extrinsics;
  extrinsics.body_p_sensor = Eigen::Vector3d(0.1, 0.2, 0.3);
  config.extrinsics = extrinsics;
  return std::make_shared<Camera>(config, name);
}

struct TestFrame {
  InputPacket packet;
  std::unique_ptr<InputData> data;
};

TestFrame makeFrame(const Sensor::ConstPtr& sensor, uint64_t stamp, uint8_t red) {
  TestFrame frame;
  frame.packet.timestamp_ns = stamp;
  frame.packet.world_t_body = Eigen::Vector3d(1.0, 2.0, 3.0);
  frame.packet.world_R_body = Eigen::Quaterniond::Identity();
  frame.packet.sensor_input = std::make_shared<ImageInputPacket>(stamp, sensor->name);
  frame.packet.sensor_input->sensor_frame = "camera_link";

  frame.data = std::make_unique<InputData>(sensor);
  frame.data->timestamp_ns = stamp;
  frame.data->world_T_body = frame.packet.world_T_body();
  // RGB order, as produced by input conversion
  frame.data->color_image = cv::Mat(3, 4, CV_8UC3, cv::Scalar(red, 20, 200));
  frame.data->depth_image = cv::Mat(3, 4, CV_32FC1, cv::Scalar(1.234f));
  frame.data->depth_image.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
  return frame;
}

std::filesystem::path makeTempDir(const std::string& name) {
  const auto path = std::filesystem::temp_directory_path() / ("hydra_obs_" + name);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::vector<std::string> readLines(const std::filesystem::path& path) {
  std::vector<std::string> lines;
  std::ifstream fin(path);
  std::string line;
  while (std::getline(fin, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

size_t countMatches(const std::vector<std::string>& lines, const std::string& token) {
  size_t count = 0;
  for (const auto& line : lines) {
    count += line.find(token) != std::string::npos ? 1 : 0;
  }
  return count;
}

}  // namespace

TEST(ObservationRecorder, DisabledDoesNothing) {
  ObservationRecorder::Config config;
  ObservationRecorder recorder(config);
  const auto camera = makeCamera("cam");
  auto frame = makeFrame(camera, 10, 100);
  EXPECT_FALSE(recorder.record(frame.packet, *frame.data, true));
  recorder.stop();
  EXPECT_EQ(recorder.stats().packets_seen, 0u);
}

TEST(ObservationRecorder, WritesEveryFrameWithCorrectColorAndDepth) {
  const auto dir = makeTempDir("write");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;
  config.queue_full_policy = "block";
  config.max_queue_frames = 2;  // force blocking to exercise the queue limits

  const auto camera = makeCamera("cam");
  {
    ObservationRecorder recorder(config);
    for (size_t i = 0; i < 20; ++i) {
      // duplicate timestamps on purpose: frame ids must still be unique
      auto frame = makeFrame(camera, 1700000000123456789ull + i / 2, 10 + i);
      EXPECT_TRUE(recorder.record(frame.packet, *frame.data, i % 3 == 0));
      // the recorder must own its copy of the images
      frame.data->color_image.setTo(cv::Scalar(0, 0, 0));
    }

    recorder.stop();
    const auto stats = recorder.stats();
    EXPECT_EQ(stats.frames_received, 20u);
    EXPECT_EQ(stats.frames_written, 20u);
    EXPECT_EQ(stats.frames_write_failed, 0u);
    EXPECT_EQ(stats.frames_dropped_queue_full, 0u);
    EXPECT_EQ(stats.depth_written, 20u);
    EXPECT_LE(stats.max_queue_frames_seen, 2u);
  }

  const auto lines = readLines(dir / "frames_raw.jsonl");
  ASSERT_EQ(lines.size(), 20u);
  EXPECT_EQ(countMatches(lines, "\"status\": \"written\""), 20u);
  // timestamps are written as exact integers
  EXPECT_NE(lines[0].find("\"timestamp_ns\": 1700000000123456789"), std::string::npos);
  EXPECT_NE(lines[0].find("\"frame_id\": 0,"), std::string::npos);
  EXPECT_NE(lines[1].find("\"frame_id\": 1,"), std::string::npos);
  EXPECT_NE(lines[0].find("\"sensor_frame\": \"camera_link\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"triggered_full_update\": true"), std::string::npos);
  EXPECT_NE(lines[1].find("\"triggered_full_update\": false"), std::string::npos);
  // world_T_sensor = world_T_body * body_T_sensor = (1.1, 2.2, 3.3)
  EXPECT_NE(lines[0].find("\"world_T_sensor\": {\"t\": [1.1000000000000001, "
                          "2.2000000000000002, 3.2999999999999998]"),
            std::string::npos)
      << lines[0];

  const cv::Mat bgr = cv::imread((dir / "rgb/00000005.png").string(), cv::IMREAD_COLOR);
  ASSERT_FALSE(bgr.empty());
  EXPECT_EQ(bgr.cols, 4);
  EXPECT_EQ(bgr.rows, 3);
  // stored as a standard png: reading with OpenCV gives BGR = (200, 20, red)
  const auto pixel = bgr.at<cv::Vec3b>(1, 1);
  EXPECT_EQ(pixel[0], 200);
  EXPECT_EQ(pixel[1], 20);
  EXPECT_EQ(pixel[2], 15);

  const cv::Mat depth =
      cv::imread((dir / "depth/00000005.tiff").string(), cv::IMREAD_UNCHANGED);
  ASSERT_FALSE(depth.empty());
  EXPECT_EQ(depth.type(), CV_32FC1);
  EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 1.234f);
  EXPECT_TRUE(std::isnan(depth.at<float>(0, 0)));

  EXPECT_TRUE(std::filesystem::exists(dir / "camera_calib.json"));
  EXPECT_TRUE(std::filesystem::exists(dir / "recorder_stats.json"));
  std::filesystem::remove_all(dir);
}

TEST(ObservationRecorder, ParallelWritersKeepMetadataOrder) {
  const auto dir = makeTempDir("parallel");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;
  config.queue_full_policy = "block";
  config.max_queue_frames = 8;
  config.num_writer_threads = 4;

  const auto camera = makeCamera("cam");
  {
    ObservationRecorder recorder(config);
    for (size_t i = 0; i < 64; ++i) {
      auto frame = makeFrame(camera, 1700000000000000000ull + i, static_cast<uint8_t>(i));
      EXPECT_TRUE(recorder.record(frame.packet, *frame.data, true));
    }

    recorder.stop();
    const auto stats = recorder.stats();
    EXPECT_EQ(stats.frames_written, 64u);
    EXPECT_EQ(stats.depth_written, 64u);
    EXPECT_EQ(stats.frames_write_failed, 0u);
  }

  const auto lines = readLines(dir / "frames_raw.jsonl");
  ASSERT_EQ(lines.size(), 64u);
  for (size_t i = 0; i < lines.size(); ++i) {
    const auto token = "\"frame_id\": " + std::to_string(i) + ",";
    EXPECT_NE(lines[i].find(token), std::string::npos) << "line " << i << ": " << lines[i];
  }

  for (size_t i = 0; i < 64; i += 21) {
    char name[16];
    std::snprintf(name, sizeof(name), "%08zu", i);
    const cv::Mat bgr =
        cv::imread((dir / "rgb" / (std::string(name) + ".png")).string(), cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());
    EXPECT_EQ(bgr.at<cv::Vec3b>(1, 1)[2], i);
  }

  std::filesystem::remove_all(dir);
}

TEST(ObservationRecorder, MillimeterDepthMasksInvalid) {
  const auto dir = makeTempDir("depth_mm");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;
  config.depth_format = "png_u16_mm";
  const auto camera = makeCamera("cam");
  {
    ObservationRecorder recorder(config);
    auto frame = makeFrame(camera, 5, 1);
    frame.data->depth_image.at<float>(0, 1) = 100.0f;  // beyond uint16 millimeters
    EXPECT_TRUE(recorder.record(frame.packet, *frame.data, false));
  }

  const cv::Mat depth =
      cv::imread((dir / "depth/00000000.png").string(), cv::IMREAD_UNCHANGED);
  ASSERT_FALSE(depth.empty());
  EXPECT_EQ(depth.type(), CV_16UC1);
  EXPECT_EQ(depth.at<uint16_t>(1, 1), 1234);
  EXPECT_EQ(depth.at<uint16_t>(0, 0), 0);
  EXPECT_EQ(depth.at<uint16_t>(0, 1), 0);
  std::filesystem::remove_all(dir);
}

TEST(ObservationRecorder, ReportsInvalidAndColorlessInputs) {
  const auto dir = makeTempDir("invalid");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;
  const auto camera = makeCamera("cam");
  ObservationRecorder recorder(config);

  auto frame = makeFrame(camera, 5, 1);
  recorder.recordInvalid(frame.packet, "input conversion failed");
  frame.data->color_image = cv::Mat();
  EXPECT_FALSE(recorder.record(frame.packet, *frame.data, false));
  recorder.stop();

  const auto stats = recorder.stats();
  EXPECT_EQ(stats.packets_seen, 2u);
  EXPECT_EQ(stats.packets_invalid, 1u);
  EXPECT_EQ(stats.packets_without_color, 1u);
  EXPECT_EQ(stats.frames_received, 0u);
  const auto lines = readLines(dir / "frames_raw.jsonl");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("\"status\": \"invalid_input\""), std::string::npos);
  EXPECT_NE(lines[0].find("input conversion failed"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(ObservationRecorder, DropPolicyReportsDroppedFrames) {
  const auto dir = makeTempDir("drop");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;
  config.queue_full_policy = "drop";
  config.max_queue_frames = 1;
  config.max_queue_bytes = 1;  // only a single frame may ever be queued

  const auto camera = makeCamera("cam");
  ObservationRecorder recorder(config);
  size_t accepted = 0;
  for (size_t i = 0; i < 200; ++i) {
    auto frame = makeFrame(camera, i, 1);
    accepted += recorder.record(frame.packet, *frame.data, false) ? 1 : 0;
  }
  recorder.stop();

  const auto stats = recorder.stats();
  EXPECT_EQ(stats.frames_received, 200u);
  EXPECT_EQ(stats.frames_enqueued, accepted);
  EXPECT_EQ(stats.frames_enqueued + stats.frames_dropped_queue_full, 200u);
  EXPECT_EQ(stats.frames_written, accepted);

  // every received frame has a metadata line, dropped ones included
  const auto lines = readLines(dir / "frames_raw.jsonl");
  EXPECT_EQ(lines.size(), 200u);
  EXPECT_EQ(countMatches(lines, "\"status\": \"dropped_queue_full\""),
            stats.frames_dropped_queue_full);
  std::filesystem::remove_all(dir);
}

TEST(ObservationRecorder, WriteFailuresAreCounted) {
  const auto dir = makeTempDir("fail");
  ObservationRecorder::Config config;
  config.enable = true;
  config.output_path = dir;

  const auto camera = makeCamera("cam");
  ObservationRecorder recorder(config);
  // replace the rgb directory with a file so image writes fail
  std::filesystem::remove_all(dir / "rgb");
  std::ofstream(dir / "rgb") << "not a directory";

  auto frame = makeFrame(camera, 1, 1);
  EXPECT_TRUE(recorder.record(frame.packet, *frame.data, false));
  recorder.stop();

  const auto stats = recorder.stats();
  EXPECT_EQ(stats.frames_write_failed, 1u);
  EXPECT_EQ(stats.frames_written, 0u);
  const auto lines = readLines(dir / "frames_raw.jsonl");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("\"status\": \"write_failed\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"rgb_path\": null"), std::string::npos);
  std::filesystem::remove_all(dir);
}

}  // namespace hydra
