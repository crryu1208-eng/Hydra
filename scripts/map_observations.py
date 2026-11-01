#!/usr/bin/env python3
"""
Map frames recorded by the ObservationRecorder to Hydra agent (trajectory) nodes.

Inputs:
  - <observations>/frames_raw.jsonl  written online by the ObservationRecorder
  - <hydra_output>/frontend/dsg.json agent nodes with their original (odometry) poses
  - <hydra_output>/backend/dsg.json  agent nodes with their optimized poses

Outputs (in <observations> unless --output is given):
  - frames.jsonl           frame metadata + trajectory mapping + corrected pose
  - trajectory_nodes.csv   agent node id, timestamp, optimized and original pose
  - agent_frames.json      reverse index: agent node -> frame ids
  - mapping_stats.json     mapping counts and graph version information

Conventions:
  - all poses are a_T_b (maps points in frame b to frame a)
  - poses are stored as {"t": [x, y, z], "q_wxyz": [w, x, y, z]}
  - "world" is the frame Hydra integrates in (the odometry frame of the input);
    "corrected" poses are in the same frame after backend optimization
  - agent nodes store the *body* pose, so camera poses use body_T_sensor

Only numpy is required. Timestamps are kept as python integers (exact nanoseconds).
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SYMBOL_INDEX_BITS = 56
SYMBOL_INDEX_MASK = (1 << SYMBOL_INDEX_BITS) - 1


# --------------------------------------------------------------------------- poses
def quat_to_rot(q_wxyz) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a rotation matrix."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> List[float]:
    """Convert a rotation matrix to a (w, x, y, z) quaternion with w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def make_pose(t, q_wxyz) -> np.ndarray:
    """Build a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(q_wxyz)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def pose_from_json(record) -> np.ndarray:
    return make_pose(record["t"], record["q_wxyz"])


def pose_to_json(T: np.ndarray):
    return {"t": [float(v) for v in T[:3, 3]], "q_wxyz": rot_to_quat(T[:3, :3])}


def invert(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def rotation_angle(T: np.ndarray) -> float:
    """Angle of the rotation part of T in radians."""
    c = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


# --------------------------------------------------------------------- graph input
@dataclass
class AgentNode:
    node_id: int
    timestamp_ns: int
    world_T_body_original: Optional[np.ndarray] = None
    world_T_body_optimized: Optional[np.ndarray] = None
    frames: List[int] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        key = self.node_id >> SYMBOL_INDEX_BITS
        index = self.node_id & SYMBOL_INDEX_MASK
        prefix = chr(key) if 0 < key < 128 and chr(key).isalpha() else ""
        return f"{prefix}{index}"


def load_agent_nodes(dsg_path: pathlib.Path, prefix: Optional[str] = None):
    """
    Read agent nodes from a spark_dsg json file.

    The json is parsed directly (instead of with the spark_dsg python bindings) so that
    timestamps keep nanosecond precision.
    """
    with dsg_path.open("r") as fin:
        record = json.load(fin)

    nodes = {}
    for node in record.get("nodes", []):
        attrs = node.get("attributes", {})
        if attrs.get("type") != "AgentNodeAttributes":
            continue

        node_id = int(node["id"])
        if prefix is not None and chr(node_id >> SYMBOL_INDEX_BITS) != prefix:
            continue

        q = attrs["world_R_body"]
        pose = make_pose(attrs["position"], [q["w"], q["x"], q["y"], q["z"]])
        nodes[node_id] = (int(attrs["timestamp"]), pose)

    return nodes, record.get("header")


def file_digest(path: pathlib.Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_loop_closures(path: pathlib.Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r") as fin:
        return max(0, sum(1 for line in fin if line.strip()) - 1)


def build_agent_nodes(original: Dict, optimized: Dict) -> Dict[int, AgentNode]:
    """Combine original and optimized agent poses by node id."""
    nodes = {}
    for node_id, (stamp, pose) in original.items():
        nodes[node_id] = AgentNode(node_id, stamp, world_T_body_original=pose)

    for node_id, (stamp, pose) in optimized.items():
        if node_id not in nodes:
            nodes[node_id] = AgentNode(node_id, stamp)
        elif nodes[node_id].timestamp_ns != stamp:
            print(
                f"[WARNING] node {node_id} has different stamps in frontend ({stamp}) "
                f"and backend ({nodes[node_id].timestamp_ns})",
                file=sys.stderr,
            )
        nodes[node_id].world_T_body_optimized = pose

    return nodes


# ------------------------------------------------------------------------- mapping
@dataclass
class MappingConfig:
    #! frames within this time of a node's stamp correspond directly to the node
    direct_tolerance_ns: int = 0
    #! max time between a frame and the node it is anchored to
    max_anchor_dt_s: float = 0.5
    #! max distance between the frame body pose and the anchor node (original poses)
    max_anchor_distance_m: float = 1.0
    #! max rotation between the frame body pose and the anchor node (original poses)
    max_anchor_angle_deg: float = 45.0
    #! tolerance for checking that a direct frame agrees with its node's pose
    direct_pose_tolerance_m: float = 1.0e-3
    direct_pose_tolerance_rad: float = 1.0e-3


def map_frame(frame, nodes_by_time, stamps, config: MappingConfig):
    """
    Find the trajectory node for one frame.

    Returns a dict describing the mapping. The relative pose node_T_body is computed
    from *original* (unoptimized) poses, which is the only consistent pair: the frame
    pose and the node pose were both measured by odometry before any loop closure.
    """
    stamp = int(frame["timestamp_ns"])
    world_T_body = pose_from_json(frame["world_T_body"])
    if not stamps:
        return {"type": "unmapped", "status": "no_agent_nodes"}

    # direct correspondence: the node was created from this exact input
    idx = bisect.bisect_left(stamps, stamp - config.direct_tolerance_ns)
    direct = []
    while idx < len(stamps) and stamps[idx] <= stamp + config.direct_tolerance_ns:
        direct.append(nodes_by_time[idx])
        idx += 1

    if direct:
        node = min(direct, key=lambda n: abs(n.timestamp_ns - stamp))
        result = {
            "type": "direct",
            "node_id": node.node_id,
            "node_symbol": node.symbol,
            "dt_ns": stamp - node.timestamp_ns,
        }
        if node.world_T_body_original is None:
            # no original pose: by construction the node is the frame's body pose
            result["node_T_body"] = pose_to_json(np.eye(4))
            result["status"] = "ok_no_original_node_pose"
            return result

        node_T_body = invert(node.world_T_body_original) @ world_T_body
        result["node_T_body"] = pose_to_json(node_T_body)
        dist = float(np.linalg.norm(node_T_body[:3, 3]))
        angle = rotation_angle(node_T_body)
        consistent = (
            dist <= config.direct_pose_tolerance_m
            and angle <= config.direct_pose_tolerance_rad
        )
        result["status"] = "ok" if consistent else "ok_pose_mismatch"
        result["distance_m"] = dist
        result["angle_rad"] = angle
        return result

    # anchor: closest node in time within the time, distance and angle limits
    max_dt_ns = int(round(config.max_anchor_dt_s * 1e9))
    lo = bisect.bisect_left(stamps, stamp - max_dt_ns)
    hi = bisect.bisect_right(stamps, stamp + max_dt_ns)
    candidates = nodes_by_time[lo:hi]
    if not candidates:
        return {"type": "unmapped", "status": "no_node_within_time"}

    max_angle = math.radians(config.max_anchor_angle_deg)
    best = None
    # when every candidate is rejected, report the check that the closest-to-passing
    # candidate failed: a node within distance but rotated beyond the limit is reported as
    # outside_rotation even if other (farther) nodes in the time window are outside_distance
    rejection_rank = {"anchor_without_original_pose": 0, "outside_distance": 1, "outside_rotation": 2}
    rejected_reason = None

    def reject(reason):
        nonlocal rejected_reason
        if rejected_reason is None or rejection_rank[reason] > rejection_rank[rejected_reason]:
            rejected_reason = reason

    for node in candidates:
        if node.world_T_body_original is None:
            reject("anchor_without_original_pose")
            continue

        node_T_body = invert(node.world_T_body_original) @ world_T_body
        dist = float(np.linalg.norm(node_T_body[:3, 3]))
        angle = rotation_angle(node_T_body)
        if dist > config.max_anchor_distance_m:
            reject("outside_distance")
            continue
        if angle > max_angle:
            reject("outside_rotation")
            continue

        key = (abs(node.timestamp_ns - stamp), dist)
        if best is None or key < best[0]:
            best = (key, node, node_T_body, dist, angle)

    if best is None:
        return {"type": "unmapped", "status": rejected_reason or "no_valid_anchor"}

    _, node, node_T_body, dist, angle = best
    return {
        "type": "anchor",
        "status": "ok",
        "node_id": node.node_id,
        "node_symbol": node.symbol,
        "dt_ns": stamp - node.timestamp_ns,
        "node_T_body": pose_to_json(node_T_body),
        "distance_m": dist,
        "angle_rad": angle,
    }


def corrected_pose(frame, mapping, nodes: Dict[int, AgentNode]):
    """
    Corrected body and sensor poses from the optimized pose of the mapped node.

    world_T_body' = world_T_node(optimized) * node_T_body
    world_T_sensor' = world_T_body' * body_T_sensor

    For anchored frames this only propagates the correction of the anchor node; any
    drift between the anchor node and the frame is not corrected.
    """
    if mapping["type"] == "unmapped":
        return None

    node = nodes[mapping["node_id"]]
    if node.world_T_body_optimized is None:
        return {"pose_type": None, "status": "node_missing_in_backend"}

    node_T_body = pose_from_json(mapping["node_T_body"])
    body_T_sensor = pose_from_json(frame["body_T_sensor"])
    world_T_body = node.world_T_body_optimized @ node_T_body
    return {
        "pose_type": (
            "optimized_direct" if mapping["type"] == "direct" else "anchor_propagated"
        ),
        "status": "ok",
        "world_T_body": pose_to_json(world_T_body),
        "world_T_sensor": pose_to_json(world_T_body @ body_T_sensor),
    }


def map_observations(frames, nodes: Dict[int, AgentNode], config: MappingConfig):
    """Map every frame and fill in the per-node frame lists."""
    nodes_by_time = sorted(nodes.values(), key=lambda n: (n.timestamp_ns, n.node_id))
    stamps = [n.timestamp_ns for n in nodes_by_time]
    results = []
    for frame in frames:
        out = dict(frame)
        if frame.get("world_T_body") is None or frame.get("status") == "invalid_input":
            out["mapping"] = {"type": "unmapped", "status": "invalid_input"}
            out["corrected"] = None
            results.append(out)
            continue

        mapping = map_frame(frame, nodes_by_time, stamps, config)
        out["mapping"] = mapping
        out["corrected"] = corrected_pose(frame, mapping, nodes)
        if mapping["type"] != "unmapped" and frame.get("rgb_path"):
            nodes[mapping["node_id"]].frames.append(int(frame["frame_id"]))
        results.append(out)

    return results


def check_integrity(results, nodes: Dict[int, AgentNode]):
    """Check that the forward and reverse indices agree. Returns a list of errors."""
    errors = []
    frame_ids = [int(r["frame_id"]) for r in results]
    if len(set(frame_ids)) != len(frame_ids):
        errors.append("duplicate frame ids")

    by_id = {int(r["frame_id"]): r for r in results}
    for node in nodes.values():
        for fid in node.frames:
            frame = by_id.get(fid)
            if frame is None:
                errors.append(f"node {node.symbol} lists unknown frame {fid}")
            elif frame["mapping"].get("node_id") != node.node_id:
                errors.append(f"frame {fid} does not point back to node {node.symbol}")

    for r in results:
        node_id = r["mapping"].get("node_id")
        if node_id is not None and node_id not in nodes:
            errors.append(f"frame {r['frame_id']} points to unknown node {node_id}")
    return errors


# ----------------------------------------------------------------------------- io
def read_jsonl(path: pathlib.Path):
    with path.open("r") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def write_outputs(output: pathlib.Path, results, nodes, stats):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "frames.jsonl").open("w") as fout:
        for r in results:
            fout.write(json.dumps(r) + "\n")

    with (output / "trajectory_nodes.csv").open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["node_id", "node_symbol", "timestamp_ns", "x", "y", "z", "qw", "qx", "qy",
             "qz", "orig_x", "orig_y", "orig_z", "orig_qw", "orig_qx", "orig_qy",
             "orig_qz", "num_frames"]
        )
        for node in sorted(nodes.values(), key=lambda n: (n.timestamp_ns, n.node_id)):
            row = [node.node_id, node.symbol, node.timestamp_ns]
            for pose in (node.world_T_body_optimized, node.world_T_body_original):
                if pose is None:
                    row += [""] * 7
                else:
                    p = pose_to_json(pose)
                    row += p["t"] + p["q_wxyz"]
            row.append(len(node.frames))
            writer.writerow(row)

    index = {
        str(node.node_id): {
            "node_symbol": node.symbol,
            "timestamp_ns": node.timestamp_ns,
            "frames": node.frames,
        }
        for node in nodes.values()
    }
    with (output / "agent_frames.json").open("w") as fout:
        json.dump(index, fout, indent=2)

    with (output / "mapping_stats.json").open("w") as fout:
        json.dump(stats, fout, indent=2)


def summarize(results, nodes, errors, graph_info):
    counts = {}
    for r in results:
        key = f"{r['mapping']['type']}:{r['mapping']['status']}"
        counts[key] = counts.get(key, 0) + 1

    status = {}
    for r in results:
        status[r.get("status")] = status.get(r.get("status"), 0) + 1

    return {
        "num_frames": len(results),
        "frame_status": status,
        "mapping": counts,
        "num_agent_nodes": len(nodes),
        "agent_nodes_with_frames": sum(1 for n in nodes.values() if n.frames),
        "agent_nodes_missing_in_backend": sum(
            1 for n in nodes.values() if n.world_T_body_optimized is None
        ),
        "integrity_errors": errors,
        "graph_version": graph_info,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("observations", type=pathlib.Path)
    parser.add_argument("hydra_output", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None)
    parser.add_argument("--prefix", default=None, help="robot prefix of agent nodes")
    parser.add_argument("--direct-tolerance-ns", type=int, default=0)
    parser.add_argument("--max-anchor-dt-s", type=float, default=0.5)
    parser.add_argument("--max-anchor-distance-m", type=float, default=1.0)
    parser.add_argument("--max-anchor-angle-deg", type=float, default=45.0)
    args = parser.parse_args(argv)

    config = MappingConfig(
        direct_tolerance_ns=args.direct_tolerance_ns,
        max_anchor_dt_s=args.max_anchor_dt_s,
        max_anchor_distance_m=args.max_anchor_distance_m,
        max_anchor_angle_deg=args.max_anchor_angle_deg,
    )

    frontend_path = args.hydra_output / "frontend" / "dsg.json"
    backend_path = args.hydra_output / "backend" / "dsg.json"
    original, _ = (
        load_agent_nodes(frontend_path, args.prefix)
        if frontend_path.exists()
        else ({}, None)
    )
    if backend_path.exists():
        optimized, header = load_agent_nodes(backend_path, args.prefix)
    else:
        print(f"[WARNING] missing {backend_path}: no corrected poses", file=sys.stderr)
        optimized, header = {}, None

    nodes = build_agent_nodes(original, optimized)
    frames = read_jsonl(args.observations / "frames_raw.jsonl")
    results = map_observations(frames, nodes, config)
    errors = check_integrity(results, nodes)

    graph_info = {
        "pose_source": str(backend_path),
        "backend_dsg_sha256": file_digest(backend_path),
        "frontend_dsg_sha256": file_digest(frontend_path),
        "dsg_header": header,
        "num_loop_closures": count_loop_closures(
            args.hydra_output / "backend" / "loop_closures.csv"
        ),
        "mapping_config": vars(config),
    }
    stats = summarize(results, nodes, errors, graph_info)
    write_outputs(args.output or args.observations, results, nodes, stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "graph_version"}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
