# Observation recording and trajectory mapping

This feature saves every valid color (and depth) observation that reaches Hydra's
reconstruction module, together with its capture pose, and links each frame to the
agent (trajectory) nodes of the scene graph. It is meant to provide images for later
offline analysis (e.g., with a VLM); it does not change scene graph construction.

## Which frames are recorded

| Scope | Recorded? |
| --- | --- |
| Frames produced by the camera driver | Not observable by Hydra |
| Frames dropped before Hydra (ROS `message_filters` sync of color/depth/labels, `input_separation_s` in the receiver, missing body pose / TF in the input module) | **No** |
| Inputs popped by `ReconstructionModule` | Counted (`packets_seen`) |
| ...that fail input conversion | Metadata line with `status: invalid_input` |
| ...without an 8-bit 3-channel color image | Counted (`packets_without_color`) |
| ...valid color frames | **Yes** (`frames_received`), with or without a mesh update |

Only full updates (`full_update_separation_s`) produce active window outputs, sinks and
agent nodes, so the recorder is called inside `ReconstructionModule::spinOnce` right
after the TSDF integration, before the early return for partial updates. Each line of
`frames_raw.jsonl` has `triggered_full_update` to tell which frames produced an output.

The queue between the input module and the active window is blocking, so nothing is
dropped between the two modules.

## Configuration

Add to the `active_window` section (the `ReconstructionModule` config):

```yaml
active_window:
  type: ReconstructionModule
  observation_recorder:
    enable: true
    output_path: /data/hydra_observations/run_01   # keep outside the hydra log_path
    queue_full_policy: block    # block: keep every frame (may stall reconstruction)
                                # drop: never stall, dropped frames are reported
    max_queue_frames: 32
    max_queue_bytes: 536870912  # 512 MiB, 0 disables the byte limit
    save_depth: true
    depth_format: tiff_f32      # lossless float meters, or png_u16_mm
    png_compression: 1
```

`hydra_node` creates the log `DataDirectory` after the pipeline stops; when
`output.overwrite` is set this can remove the contents of `log_path`, so use a separate
`output_path` for observations.

Writing happens on a background thread. Images are deep-copied before being queued, the
queue is bounded by frame count and bytes, and `stop()` (called from
`ReconstructionModule::stop()` and the destructor) writes every queued frame before
writing `camera_calib.json` and `recorder_stats.json`. Image files are written to a
temporary name and renamed, so a failed write never leaves a partial file under the
final name. Every failure is logged at `ERROR` and counted.

## Online output (`output_path`)

```
rgb/00000000.png          standard PNG (reads as BGR with cv2.imread, RGB with PIL)
depth/00000000.tiff       32-bit float meters (NaN/inf kept), or 16-bit mm PNG
frames_raw.jsonl          one line per frame (written, write_failed, dropped, invalid)
camera_calib.json         per sensor: pinhole intrinsics, ranges, body_T_sensor
recorder_stats.json       counters
```

`frames_raw.jsonl` line:

```json
{"frame_id": 12, "status": "written", "sensor": "camera", "sensor_frame": "camera_link",
 "sensor_sequence": 12, "timestamp_ns": 1700000000123456789,
 "timestamp_ns_str": "1700000000123456789", "rgb_path": "rgb/00000012.png",
 "depth_path": "depth/00000012.tiff", "width": 640, "height": 480, "has_depth": true,
 "triggered_full_update": false,
 "world_T_body": {"t": [x, y, z], "q_wxyz": [w, x, y, z]},
 "body_T_sensor": {...}, "world_T_sensor": {...}, "error": null}
```

- `frame_id` is unique per recorder run (timestamps are not unique across sensors).
- `timestamp_ns` is the input stamp (header stamp of the color image for ROS inputs),
  written as an exact integer; `timestamp_ns_str` is for readers that parse numbers as
  doubles.
- Color in `InputData` is RGB (ROS images are converted with `cv_bridge` to `rgb8`;
  `color_is_bgr` packets are converted by `ImageInputPacket`). The recorder converts to
  BGR only for `cv::imwrite`, so the PNG has the correct colors.
- `world` is the frame Hydra integrates in (the odometry frame of the body pose lookup).
  `world_T_sensor = world_T_body * body_T_sensor`.
- Hydra's `Camera` model has no distortion parameters; inputs are expected to be
  rectified. `camera_calib.json` states this explicitly.

## Trajectory mapping (`scripts/map_observations.py`)

Run after Hydra has saved its output:

```bash
python3 scripts/map_observations.py /data/hydra_observations/run_01 <hydra log_path>
```

It reads `frontend/dsg.json` (agent nodes with original odometry poses) and
`backend/dsg.json` (agent nodes with optimized poses). The json is parsed directly so
nanosecond timestamps are exact. Only numpy is needed.

How agent nodes are made (Hydra `main`): `GraphBuilder::processNextInput` calls
`PoseGraphFromOdom::update(msg.timestamp_ns, world_T_body)` for every active window
output, and nodes closer than `min_time_separation_s` / `min_pose_separation` are
skipped. A node's timestamp is therefore exactly the stamp of the input that created it,
and there are fewer nodes than frames.

Mapping policy for each frame:

1. **direct**: a node with `|t_node - t_frame| <= direct_tolerance_ns` (default 0).
   `node_T_body` is checked against the frame pose; a disagreement is reported as
   `ok_pose_mismatch` (e.g., external pose graphs).
2. **anchor**: otherwise, the node closest in time with `|dt| <= max_anchor_dt_s`
   (0.5 s), distance `<= max_anchor_distance_m` (1 m) and rotation
   `<= max_anchor_angle_deg` (45 deg), all measured between *original* poses. Ties go to
   the earlier node.
3. **unmapped**: none of the above (`no_node_within_time`, `outside_distance`,
   `outside_rotation`, `no_agent_nodes`, `invalid_input`). Frames are never forced onto
   the nearest node.

Corrected poses after loop closure:

```
node_T_body        = inverse(world_T_node_original) * world_T_body_capture
world_T_body_corr  = world_T_node_optimized * node_T_body
world_T_sensor_corr = world_T_body_corr * body_T_sensor
```

`pose_type` is `optimized_direct` for direct frames and `anchor_propagated` for anchored
frames. Anchored frames only get the correction of their anchor node; drift between the
anchor and the frame is not corrected. The capture pose is never modified.

Node changes: a node present in the frontend but missing from the backend graph keeps its
mapping and gets `corrected.status = node_missing_in_backend`. Each run of the tool
records the sha256 of both graphs, the number of loop closures and the mapping limits in
`mapping_stats.json`, so corrected poses can be traced to the graph version they came
from. Re-run the tool after a new graph is saved.

Outputs (next to `frames_raw.jsonl` by default):

```
frames.jsonl          frames_raw fields + "mapping" + "corrected"
trajectory_nodes.csv  node_id,node_symbol,timestamp_ns,optimized pose,original pose,num_frames
agent_frames.json     node_id -> {node_symbol, timestamp_ns, frames: [frame_id, ...]}
mapping_stats.json    counts per mapping status, integrity errors, graph version
```

`backend/trajectory.csv` is not modified. The tool exits with an error code when the
forward (`frame -> node`) and reverse (`node -> frames`) indices disagree.

## Not implemented

Frame to object/place observation links are not written (`observations.jsonl`). Hydra
`main` has no per-frame record of which objects were seen: objects come from mesh
clustering over the active window, not from individual images. A reliable link needs
projection of the object geometry into each saved frame plus an occlusion test against
the saved depth, which can be done offline with the files above; it is not done here.

## Tests

- `tests/utils/test_observation_recorder.cpp` (gtest): all frames written, unique ids
  with duplicate timestamps, image ownership, color order, float/mm depth, invalid and
  colorless inputs, queue-full drop accounting, write failures.
- `scripts/test_map_observations.py` (pytest): quaternion/transform math, direct and
  anchor mapping, limits, multiple sensors, pose mismatch, loop-closure correction for
  direct and anchored frames, deleted nodes, end-to-end file round trip.
