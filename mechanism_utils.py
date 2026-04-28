# mechanism_utils.py
import json
import os
import sys

import numpy as np


def load_mechanism_from_json(json_filename):
    """Load a mechanism JSON file and convert joint data to screw coordinates.

    Data convention:
    - ``axis`` stores the joint motion direction for both R and P joints.
    - ``pos`` stores the joint position or geometric reference point.
    - For an R joint, screw = [w; r x w / L_char].
    - For a P joint, screw = [0; v].
    """
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)

    file_path = os.path.join(base_dir, "mechanisms", json_filename)
    if not file_path.endswith(".json"):
        file_path += ".json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_screw_map = {}
    nodes_info = {}

    link_lengths = []
    nodes_dict = {n["id"]: np.array(n["pos"], dtype=np.float64) for n in data["nodes"]}
    for u, v in data["links"]:
        dist = np.linalg.norm(nodes_dict[u] - nodes_dict[v])
        if dist > 1e-6:
            link_lengths.append(dist)

    L_char = float(np.mean(link_lengths)) if link_lengths else 1.0
    if L_char < 1e-9:
        L_char = 1.0
    print(f"📏 检测到机构特征长度 L_char = {L_char:.2f}")

    for node in data["nodes"]:
        nid = node["id"]
        j_type = node["type"].upper()
        vec_axis = np.array(node["axis"], dtype=np.float64)
        vec_pos = np.array(node["pos"], dtype=np.float64)
        axis_norm = np.linalg.norm(vec_axis)
        screw = np.zeros(6, dtype=np.float64)

        if axis_norm <= 1e-9:
            raise ValueError(
                f"{j_type} joint {nid} has a zero axis. "
                "Store the joint motion direction in the axis field."
            )

        direction = vec_axis / axis_norm
        if j_type == "R":
            screw[:3] = direction
            screw[3:] = np.cross(vec_pos, direction) / L_char
        elif j_type == "P":
            screw[:3] = 0.0
            screw[3:] = direction
        else:
            raise ValueError(f"Unsupported joint type {j_type!r} at node {nid}.")

        node_screw_map[nid] = screw
        nodes_info[nid] = {
            "type": j_type,
            "axis": vec_axis,
            "pos": vec_pos,
            "screw": screw,
        }

    links = [tuple(link) for link in data["links"]]
    rigid_body_sets = [set(rb) for rb in data.get("rigid_bodies", [])]
    settings = data.get("settings", {})

    manual_path = settings.get("manual_path", None)
    base_node = settings.get("base_node", None)
    ee_node = settings.get("ee_node", None)
    base_link = settings.get("base_link", None)
    ee_link = settings.get("ee_link", None)

    if base_node is None and manual_path is None and base_link is None:
        base_node = 0
    if ee_node is None and manual_path is None and ee_link is None:
        ee_node = len(data["nodes"]) - 1

    return (
        node_screw_map,
        links,
        base_node,
        ee_node,
        manual_path,
        nodes_info,
        rigid_body_sets,
        base_link,
        ee_link,
    )
