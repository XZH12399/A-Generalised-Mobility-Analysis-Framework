# mechanism_utils.py
import json
import numpy as np
import os
import sys


def load_mechanism_from_json(json_filename):
    """
    读取 JSON 文件并解析为 dof_analysis 需要的 NumPy 格式。
    注意：为了兼容旧版逻辑，P副方向依然使用 pos 计算。
    """
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)

    file_path = os.path.join(base_dir, 'mechanisms', json_filename)
    if not file_path.endswith('.json'): file_path += '.json'
    if not os.path.exists(file_path): raise FileNotFoundError(f"Missing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    node_screw_map = {}
    nodes_info = {}

    # 1. 计算特征长度 L_char
    link_lengths = []
    nodes_dict = {n['id']: np.array(n['pos']) for n in data['nodes']}

    for link in data['links']:
        u, v = link
        dist = np.linalg.norm(nodes_dict[u] - nodes_dict[v])
        # 过滤掉 0 长度的虚拟杆（如果有的话）
        if dist > 1e-6:
            link_lengths.append(dist)

    # 如果没有杆长（比如纯球机构），默认设为 1.0
    if not link_lengths:
        L_char = 1.0
    else:
        # 使用平均值
        L_char = np.mean(link_lengths)
        # 或者使用中位数
        # L_char = np.median(link_lengths)

    print(f"📏 检测到机构特征长度 L_char = {L_char:.2f}")

    for node in data['nodes']:
        nid = node['id']
        j_type = node['type'].upper()
        vec_axis = np.array(node['axis'], dtype=np.float64)
        vec_pos = np.array(node['pos'], dtype=np.float64)
        screw = np.zeros(6, dtype=np.float64)

        # 归一化，防止除零
        axis_norm = np.linalg.norm(vec_axis)
        pos_norm = np.linalg.norm(vec_pos)

        # 为了兼容性，保留你原始代码的逻辑：
        # R副用 axis, P副用 pos (虽然非标准，但能复现之前的 Rank)
        if j_type == 'R':
            w = vec_axis / (axis_norm + 1e-9)
            v = np.cross(vec_pos, w)

            # [核心修改] R副：线速度除以特征长度
            screw[:3] = w
            screw[3:] = v / L_char

        elif j_type == 'P':
            move_dir = vec_pos / (pos_norm + 1e-9)

            # [核心修改] P副：保持原样 (模长为1)
            # 这样 R 副模长 ~1.4, P 副模长 = 1.0, 量级完美平衡
            screw[:3] = 0.0
            screw[3:] = move_dir

        node_screw_map[nid] = screw

        # 存储完整信息用于微扰
        nodes_info[nid] = {
            'type': j_type,
            'axis': vec_axis,
            'pos': vec_pos,
            'screw': screw
        }

    links = [tuple(link) for link in data['links']]
    raw_rigid_bodies = data.get('rigid_bodies', [])
    rigid_body_sets = [set(rb) for rb in raw_rigid_bodies]
    settings = data.get('settings', {})

    manual_path = settings.get('manual_path', None)
    base_node = settings.get('base_node', None)
    ee_node = settings.get('ee_node', None)

    # [新增] 解析 Link 定义
    base_link = settings.get('base_link', None)
    ee_link = settings.get('ee_link', None)

    # 兼容性处理：如果只给了 link 没给 node，也不需要在这里强制报错，
    # 留给 dof_analysis.py 里的智能路径构建去处理。

    if base_node is None and manual_path is None and base_link is None:
        base_node = 0
    if ee_node is None and manual_path is None and ee_link is None:
        ee_node = len(data['nodes']) - 1

    return node_screw_map, links, base_node, ee_node, manual_path, nodes_info, rigid_body_sets, base_link, ee_link