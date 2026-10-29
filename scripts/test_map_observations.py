"""Tests for map_observations.py (run with `pytest scripts/test_map_observations.py`)."""

import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "map_observations", pathlib.Path(__file__).parent / "map_observations.py"
)
mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mo)

BASE_NS = 1_700_000_000_123_456_789  # beyond float64 integer precision


def rot_z(angle):
    T = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def trans(x, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


BODY_T_SENSOR = trans(0.1, 0.0, 0.2) @ mo.make_pose([0, 0, 0], [0.5, -0.5, 0.5, -0.5])


def make_frame(frame_id, stamp, world_T_body, body_T_sensor=BODY_T_SENSOR):
    return {
        "frame_id": frame_id,
        "status": "written",
        "sensor": "cam",
        "timestamp_ns": stamp,
        "rgb_path": f"rgb/{frame_id:08d}.png",
        "world_T_body": mo.pose_to_json(world_T_body),
        "body_T_sensor": mo.pose_to_json(body_T_sensor),
        "world_T_sensor": mo.pose_to_json(world_T_body @ body_T_sensor),
    }


def symbol(key, index):
    return (ord(key) << 56) | index


# --------------------------------------------------------------------------- math
def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = mo.quat_to_rot(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        q2 = mo.rot_to_quat(R)
        assert np.allclose(mo.quat_to_rot(q2), R, atol=1e-12)


def test_invert_and_angle():
    T = trans(1, 2, 3) @ rot_z(0.3)
    assert np.allclose(mo.invert(T) @ T, np.eye(4))
    assert math.isclose(mo.rotation_angle(T), 0.3, abs_tol=1e-12)


def test_symbol_decoding():
    node = mo.AgentNode(symbol("a", 42), 0)
    assert node.symbol == "a42"


# ------------------------------------------------------------------------ mapping
def make_nodes(poses, stamps, key="a"):
    nodes = {}
    for i, (pose, stamp) in enumerate(zip(poses, stamps)):
        nid = symbol(key, i)
        nodes[nid] = mo.AgentNode(nid, stamp, world_T_body_original=pose,
                                  world_T_body_optimized=pose.copy())
    return nodes


def test_direct_and_anchor_mapping():
    # nodes every 1 s along x, frames every 0.25 s; frames on node stamps are direct
    node_poses = [trans(float(i)) for i in range(4)]
    node_stamps = [BASE_NS + i * 1_000_000_000 for i in range(4)]
    nodes = make_nodes(node_poses, node_stamps)
    frames = [
        make_frame(i, BASE_NS + i * 250_000_000, trans(0.25 * i)) for i in range(13)
    ]
    results = mo.map_observations(frames, nodes, mo.MappingConfig())

    for r in results:
        i = r["frame_id"]
        if i % 4 == 0:
            assert r["mapping"]["type"] == "direct"
            assert r["mapping"]["status"] == "ok"
            assert r["mapping"]["dt_ns"] == 0
            assert r["mapping"]["node_id"] == symbol("a", i // 4)
        else:
            assert r["mapping"]["type"] == "anchor"
            # nearest node in time: 0.25 -> node 0, 0.75 -> node 1, 0.5 -> tie -> node 0
            expected = round(i / 4 - 0.01)
            assert r["mapping"]["node_id"] == symbol("a", expected), i

    # forward / reverse index consistency
    assert mo.check_integrity(results, nodes) == []
    assert sum(len(n.frames) for n in nodes.values()) == len(frames)


def test_unmapped_outside_limits():
    nodes = make_nodes([trans(0.0)], [BASE_NS])
    config = mo.MappingConfig(max_anchor_dt_s=0.5, max_anchor_distance_m=1.0,
                              max_anchor_angle_deg=30.0)
    frames = [
        make_frame(0, BASE_NS + 600_000_000, trans(0.1)),  # too late
        make_frame(1, BASE_NS + 100_000_000, trans(1.5)),  # too far
        make_frame(2, BASE_NS + 100_000_000, rot_z(math.radians(40))),  # too rotated
        make_frame(3, BASE_NS + 100_000_000, trans(0.9) @ rot_z(math.radians(20))),
    ]
    results = mo.map_observations(frames, nodes, config)
    assert results[0]["mapping"] == {"type": "unmapped", "status": "no_node_within_time"}
    assert results[1]["mapping"] == {"type": "unmapped", "status": "outside_distance"}
    assert results[2]["mapping"] == {"type": "unmapped", "status": "outside_rotation"}
    assert results[3]["mapping"]["type"] == "anchor"
    for r in results[:3]:
        assert r["corrected"] is None
    assert nodes[symbol("a", 0)].frames == [3]


def test_no_nodes_and_invalid_frames():
    frames = [make_frame(0, BASE_NS, trans(0.0)),
              {"frame_id": 1, "status": "invalid_input", "world_T_body": None,
               "timestamp_ns": BASE_NS}]
    results = mo.map_observations(frames, {}, mo.MappingConfig())
    assert results[0]["mapping"]["status"] == "no_agent_nodes"
    assert results[1]["mapping"]["status"] == "invalid_input"


def test_duplicate_timestamps_multiple_sensors():
    nodes = make_nodes([trans(0.0)], [BASE_NS])
    other_extrinsic = trans(0.0, 0.3, 0.0)
    frames = [make_frame(0, BASE_NS, trans(0.0)),
              make_frame(1, BASE_NS, trans(0.0), body_T_sensor=other_extrinsic)]
    frames[1]["sensor"] = "cam2"
    results = mo.map_observations(frames, nodes, mo.MappingConfig())
    assert [r["mapping"]["type"] for r in results] == ["direct", "direct"]
    assert nodes[symbol("a", 0)].frames == [0, 1]
    # each sensor keeps its own extrinsic in the corrected pose
    s0 = mo.pose_from_json(results[0]["corrected"]["world_T_sensor"])
    s1 = mo.pose_from_json(results[1]["corrected"]["world_T_sensor"])
    assert np.allclose(s0, BODY_T_SENSOR)
    assert np.allclose(s1, other_extrinsic)


def test_direct_pose_mismatch_flagged():
    nodes = make_nodes([trans(0.0)], [BASE_NS])
    results = mo.map_observations([make_frame(0, BASE_NS, trans(0.05))], nodes,
                                  mo.MappingConfig())
    assert results[0]["mapping"]["type"] == "direct"
    assert results[0]["mapping"]["status"] == "ok_pose_mismatch"


# ------------------------------------------------------------------ loop closure
def test_loop_closure_correction():
    """Optimized node poses move rigidly; corrected camera poses must follow."""
    node_poses = [trans(float(i)) @ rot_z(0.1 * i) for i in range(5)]
    node_stamps = [BASE_NS + i * 1_000_000_000 for i in range(5)]
    nodes = make_nodes(node_poses, node_stamps)

    # loop closure shifts and rotates node 3 and 4 differently than the rest
    correction = trans(0.0, 0.5, 0.0) @ rot_z(0.2)
    for i in (3, 4):
        n = nodes[symbol("a", i)]
        n.world_T_body_optimized = correction @ n.world_T_body_original

    # a direct frame on node 3 and an anchored frame 0.2 s after node 3
    anchored_body = node_poses[3] @ trans(0.2) @ rot_z(0.05)
    frames = [make_frame(0, node_stamps[3], node_poses[3]),
              make_frame(1, node_stamps[3] + 200_000_000, anchored_body),
              make_frame(2, node_stamps[1], node_poses[1])]
    results = mo.map_observations(frames, nodes, mo.MappingConfig())

    c0 = results[0]["corrected"]
    assert c0["pose_type"] == "optimized_direct"
    assert np.allclose(mo.pose_from_json(c0["world_T_sensor"]),
                       correction @ node_poses[3] @ BODY_T_SENSOR, atol=1e-9)

    c1 = results[1]["corrected"]
    assert c1["pose_type"] == "anchor_propagated"
    assert np.allclose(mo.pose_from_json(c1["world_T_body"]),
                       correction @ anchored_body, atol=1e-9)
    assert np.allclose(mo.pose_from_json(c1["world_T_sensor"]),
                       correction @ anchored_body @ BODY_T_SENSOR, atol=1e-9)

    # unaffected node: corrected equals original
    c2 = results[2]["corrected"]
    assert np.allclose(mo.pose_from_json(c2["world_T_body"]), node_poses[1], atol=1e-9)

    # the original capture pose is preserved untouched
    assert results[1]["world_T_body"] == frames[1]["world_T_body"]


def test_node_missing_in_backend():
    nodes = make_nodes([trans(0.0)], [BASE_NS])
    nodes[symbol("a", 0)].world_T_body_optimized = None
    results = mo.map_observations([make_frame(0, BASE_NS, trans(0.0))], nodes,
                                  mo.MappingConfig())
    assert results[0]["mapping"]["type"] == "direct"
    assert results[0]["corrected"] == {"pose_type": None,
                                       "status": "node_missing_in_backend"}


# ----------------------------------------------------------------- end to end io
def write_dsg(path, nodes, key="a"):
    record = {"header": {"version": "test"}, "nodes": [], "edges": []}
    for index, (stamp, pose) in enumerate(nodes):
        p = mo.pose_to_json(pose)
        w, x, y, z = p["q_wxyz"]
        record["nodes"].append({
            "id": symbol(key, index),
            "layer": 2,
            "partition": ord(key),
            "attributes": {"type": "AgentNodeAttributes", "position": p["t"],
                           "timestamp": stamp,
                           "world_R_body": {"w": w, "x": x, "y": y, "z": z},
                           "external_key": 0},
        })
    # a non-agent node that must be ignored
    record["nodes"].append({"id": 7, "layer": 1, "partition": 0,
                            "attributes": {"type": "ObjectNodeAttributes"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_end_to_end(tmp_path):
    obs = tmp_path / "observations"
    out = tmp_path / "hydra"
    obs.mkdir()
    original = [(BASE_NS + i * 1_000_000_000, trans(float(i))) for i in range(3)]
    correction = trans(0.0, 0.0, 0.3)
    optimized = [(s, correction @ p) for s, p in original[:2]]  # node 2 deleted
    write_dsg(out / "frontend" / "dsg.json", original)
    write_dsg(out / "backend" / "dsg.json", optimized)
    (out / "backend" / "loop_closures.csv").write_text("header\n1,2\n")

    frames = [make_frame(i, BASE_NS + i * 500_000_000, trans(0.5 * i)) for i in range(5)]
    frames.append({"frame_id": 5, "status": "dropped_queue_full", "sensor": "cam",
                   "timestamp_ns": BASE_NS + 1, "rgb_path": None,
                   "world_T_body": mo.pose_to_json(trans(0.0)),
                   "body_T_sensor": mo.pose_to_json(BODY_T_SENSOR)})
    with (obs / "frames_raw.jsonl").open("w") as fout:
        for f in frames:
            fout.write(json.dumps(f) + "\n")

    assert mo.main([str(obs), str(out)]) == 0
    results = [json.loads(line) for line in (obs / "frames.jsonl").read_text().splitlines()]
    assert len(results) == 6
    # exact nanosecond timestamps survive the round trip
    assert results[4]["timestamp_ns"] == BASE_NS + 2_000_000_000
    assert results[4]["mapping"]["type"] == "direct"
    assert results[4]["corrected"]["status"] == "node_missing_in_backend"
    assert results[2]["corrected"]["pose_type"] == "optimized_direct"

    index = json.loads((obs / "agent_frames.json").read_text())
    # 1.0 s is direct; 1.5 s is equidistant from nodes 1 and 2 and goes to the earlier
    assert index[str(symbol("a", 1))]["frames"] == [2, 3]
    # the dropped frame is mapped but has no image so it is not indexed
    assert 5 not in index[str(symbol("a", 0))]["frames"]

    stats = json.loads((obs / "mapping_stats.json").read_text())
    assert stats["integrity_errors"] == []
    assert stats["agent_nodes_missing_in_backend"] == 1
    assert stats["graph_version"]["num_loop_closures"] == 1

    rows = (obs / "trajectory_nodes.csv").read_text().splitlines()
    assert len(rows) == 4
    assert rows[1].startswith(f"{symbol('a', 0)},a0,{BASE_NS},")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
