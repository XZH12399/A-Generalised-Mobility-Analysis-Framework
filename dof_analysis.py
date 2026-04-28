# dof_analysis.py
import networkx as nx
import numpy as np


# ==========================================
# 1. 数学辅助函数
# ==========================================

def _lie_bracket(twist1, twist2):
    w1, v1 = twist1[:3], twist1[3:]
    w2, v2 = twist2[:3], twist2[3:]
    w_new = np.cross(w1, w2)
    v_new = np.cross(w1, v2) - np.cross(w2, v1)
    return np.concatenate([w_new, v_new])


def _build_extended_path_nx(G, raw_path):
    """旧版路径扩展辅助函数（保留以兼容手动路径）"""
    if not raw_path: return None
    start, end = raw_path[0], raw_path[-1]
    path_set = set(raw_path)
    ghost_prev, ghost_next = None, None
    try:
        nbrs = list(G.neighbors(start))
        valid = [n for n in nbrs if n not in path_set]
        if valid: ghost_prev = min(valid)
    except:
        pass
    try:
        nbrs = list(G.neighbors(end))
        valid = [n for n in nbrs if n not in path_set]
        if valid: ghost_next = max(valid)
    except:
        pass
    return [ghost_prev] + raw_path + [ghost_next]


def construct_smart_path(topology_edges, base_link_str, ee_link_str):
    """
    智能路径构建：支持 "杆件-杆件"、"杆件-节点"、"节点-节点" 的任意组合。
    如果输入是杆件 "u_v"，会自动切断 u-v 边以确定延伸方向。

    Returns:
        full_path: list (包含 ghost nodes)
        start_node: int
        end_node: int
    """
    G_full = nx.Graph()
    G_full.add_edges_from(topology_edges)

    # --- 内部辅助函数：解析节点或杆件选项 ---
    def parse_anchor_opts(anchor_str):
        if anchor_str is None:
            return [], None

        s = str(anchor_str)
        if '_' in s:
            # 如果是杆件 "A_B" -> 需要把字符串转为 int ID
            u_str, v_str = s.split('_')
            try:
                u, v = int(u_str), int(v_str)
                # 选项1: 从 A 出发 (Ghost是 B)
                # 选项2: 从 B 出发 (Ghost是 A)
                return [
                    {'node': u, 'ghost': v},
                    {'node': v, 'ghost': u}
                ], (u, v)
            except ValueError:
                return [], None
        else:
            # 如果是单节点 "A"
            try:
                node_id = int(s)
                # 选项: 从 A 出发 (Ghost是 None，留给后续自动处理或留空)
                return [{'node': node_id, 'ghost': None}], None
            except ValueError:
                return [], None

    # 1. 解析基座和末端
    base_opts, base_edge_to_cut = parse_anchor_opts(base_link_str)
    ee_opts, ee_edge_to_cut = parse_anchor_opts(ee_link_str)

    if not base_opts or not ee_opts:
        return None, None, None

    # 2. 构建切割图 (G_cut)
    # 如果定义了杆件，必须切断杆件内部连接，强迫路径向外寻找
    G_cut = G_full.copy()

    if base_edge_to_cut and G_cut.has_edge(*base_edge_to_cut):
        G_cut.remove_edge(*base_edge_to_cut)

    if ee_edge_to_cut and G_cut.has_edge(*ee_edge_to_cut):
        G_cut.remove_edge(*ee_edge_to_cut)

    # 3. 组合寻找最短路径
    for b_opt in base_opts:
        for e_opt in ee_opts:
            try:
                # 在切断了内部连接的图中寻找路径
                path = nx.shortest_path(G_cut, source=b_opt['node'], target=e_opt['node'])

                # 路径构建成功！
                # 组装完整路径: [BaseGhost, Start, ..., End, EEGhost]
                full_path = [b_opt['ghost']] + path + [e_opt['ghost']]

                # 如果 Ghost 为 None (说明输入的是节点)，这里尝试自动补全一下 Ghost
                # 以兼容旧的逻辑，或者保持 None
                # 这里为了稳健性，若为 None，尝试用原来的 _build_extended_path_nx 的逻辑补一个
                if full_path[0] is None:
                    # 尝试找一个非 path 的邻居
                    nbrs = list(G_cut.neighbors(path[0]))
                    valid = [n for n in nbrs if n not in path]
                    if valid: full_path[0] = valid[0]

                if full_path[-1] is None:
                    nbrs = list(G_cut.neighbors(path[-1]))
                    valid = [n for n in nbrs if n not in path]
                    if valid: full_path[-1] = valid[0]

                return full_path, path[0], path[-1]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return None, None, None


def augment_k_matrix_to_remove_modes(K, bad_modes, weight=10.0):
    if not bad_modes: return K
    rows_to_add = []
    for mode in bad_modes:
        mode_norm = mode / (np.linalg.norm(mode) + 1e-12)
        rows_to_add.append(mode_norm * weight)
    return np.vstack((K, np.array(rows_to_add)))


def detect_instantaneous_modes(K_func_builder, candidate_modes, loops, edge_to_col, node_screw_map):
    dt = 1e-3
    idof_vectors = []
    print(f"   🕵️  正在进行多闭环漂移投影检测 (Multi-loop Drift Projection, Step={dt})...")

    K_curr = K_func_builder(node_screw_map)
    K_pinv = np.linalg.pinv(K_curr, rcond=1e-3)

    for i, mode_vec in enumerate(candidate_modes):
        mode_vec = mode_vec / (np.linalg.norm(mode_vec) + 1e-9)
        loop_drifts_list = []

        for loop in loops:
            loop_drift = np.zeros(6)
            L = len(loop)
            current_twist_sum = np.zeros(6)

            for j in range(L):
                curr_node = loop[j]
                prev_node = loop[(j - 1 + L) % L]
                next_node = loop[(j + 1) % L]

                val_next = mode_vec[edge_to_col.get((curr_node, next_node), -1)] if (curr_node,
                                                                                     next_node) in edge_to_col else 0.0
                val_prev = mode_vec[edge_to_col.get((curr_node, prev_node), -1)] if (curr_node,
                                                                                     prev_node) in edge_to_col else 0.0
                d_theta = (val_next - val_prev) * dt

                screw = node_screw_map[curr_node]
                drift_contribution = _lie_bracket(current_twist_sum, screw)
                loop_drift += drift_contribution * d_theta
                current_twist_sum += screw * d_theta

            loop_drifts_list.append(loop_drift)

        full_drift_vector = np.concatenate(loop_drifts_list)
        solution = K_pinv @ full_drift_vector
        projected_drift = K_curr @ solution
        residual_vec = full_drift_vector - projected_drift
        residual_norm = np.linalg.norm(residual_vec)
        drift_norm = np.linalg.norm(full_drift_vector)

        if drift_norm < 1e-12:
            ratio = 0.0
        else:
            ratio = residual_norm / drift_norm

        if ratio > 0.1:
            print(f"      -> Mode {i + 1}: Drift无法补偿 (Ratio={ratio:.2f}) (⚠️ IDOF)")
            idof_vectors.append(mode_vec)
        else:
            print(f"      -> Mode {i + 1}: Drift可吸收 (Ratio={ratio:.2f}) (✅ Valid)")

    return idof_vectors


# ==========================================
# 2. 主分析入口
# ==========================================

def analyze_mobility_anchor(node_screw_map, topology_edges, nodes_info,
                            rigid_body_sets=None,
                            base_node=None, ee_node=None,
                            base_link=None, ee_link=None,
                            manual_extended_path=None,
                            dof_threshold=1e-4):
    """
    Args:
        base_link (str): 格式 "u_v" (杆件) 或 "u" (节点)
        ee_link (str): 格式 "u_v" (杆件) 或 "u" (节点)
    """

    # --- 0. 拓扑与路径构建 ---
    G_raw = nx.Graph()
    for u, v in topology_edges: G_raw.add_edge(u, v)

    # 优先级 1: 用户指定了 manual_path
    extended_path = manual_extended_path
    if extended_path is not None:
        if len(extended_path) >= 4:
            print(
                f"🛤️  使用手动扩展路径: {extended_path} "
                f"(Base Link {extended_path[0]}-{extended_path[1]} -> "
                f"EE Link {extended_path[-2]}-{extended_path[-1]})"
            )
        else:
            print(f"🛤️  使用手动路径: {extended_path}")

    # 优先级 2: 用户指定了 Link 字符串 (智能路径)
    if extended_path is None and base_link and ee_link:
        print(f"🛤️  正在使用智能路径分析: Base({base_link}) -> EE({ee_link})")
        smart_path, s_node, e_node = construct_smart_path(topology_edges, base_link, ee_link)
        if smart_path:
            extended_path = smart_path
            base_node = s_node
            ee_node = e_node
            print(f"    -> 自动规划路径: {extended_path}")
        else:
            print("    ⚠️  警告: 无法构建智能路径，回退到默认逻辑。")

    # 优先级 3: 仅有 base_node / ee_node (回退到旧逻辑)
    if extended_path is None:
        if base_node is not None and ee_node is not None:
            # 旧逻辑补丁
            pass
        else:
            return {"error": "Args missing: Need (base_link, ee_link) OR (base_node, ee_node) OR manual_path"}

    # 再次确认 base/ee (用于后续 loop 检测等无关路径的逻辑)
    if base_node is None and extended_path: base_node = extended_path[1] if len(extended_path) > 1 else extended_path[0]
    if ee_node is None and extended_path: ee_node = extended_path[-2] if len(extended_path) > 1 else extended_path[-1]

    try:
        loops = nx.cycle_basis(G_raw)
    except:
        loops = []

    loop_nodes_set = set()
    loop_edges_set = set()
    if len(loops) > 0:
        for loop in loops:
            L = len(loop)
            for i in range(L):
                u, v = loop[i], loop[(i + 1) % L]
                loop_nodes_set.add(u);
                loop_nodes_set.add(v)
                loop_edges_set.add(tuple(sorted((u, v))))
    else:
        for u, v in topology_edges:
            loop_nodes_set.add(u);
            loop_nodes_set.add(v)
            loop_edges_set.add(tuple(sorted((u, v))))

    directed_edges = []
    for u, v in loop_edges_set:
        directed_edges.append((u, v));
        directed_edges.append((v, u))

    edge_to_col = {edge: i for i, edge in enumerate(directed_edges)}
    num_vars = len(directed_edges)
    num_loops = len(loops)
    num_nodes = len(loop_nodes_set)
    gauge_n = num_nodes

    # --- 闭包 ---
    def build_K_matrix(current_screw_map):
        if num_loops == 0: return np.zeros((6, num_vars))
        K_local = np.zeros((6 * num_loops, num_vars), dtype=np.float64)
        ortho_basis = np.eye(6, dtype=np.float64)
        for l_idx, loop_nodes in enumerate(loops):
            L = len(loop_nodes)
            row_start = l_idx * 6

            current_loop_set = set(loop_nodes)
            is_rigid = False
            for rb_set in rigid_body_sets:
                if current_loop_set == rb_set:
                    is_rigid = True
                    break

            for i in range(L):
                curr = loop_nodes[i]
                next_node = loop_nodes[(i + 1) % L]
                prev_node = loop_nodes[(i - 1 + L) % L]
                screw = ortho_basis[i % 6] if is_rigid else current_screw_map[curr]
                if (curr, next_node) in edge_to_col:
                    K_local[row_start:row_start + 6, edge_to_col[(curr, next_node)]] += screw
                if (curr, prev_node) in edge_to_col:
                    K_local[row_start:row_start + 6, edge_to_col[(curr, prev_node)]] -= screw
        return K_local

    # --- Phase 1 ---
    print("🔄 [Phase 1] 初始 SVD 分析...")
    K_initial = build_K_matrix(node_screw_map)
    U, S_init, Vh = np.linalg.svd(K_initial)
    full_S = np.zeros(num_vars)
    full_S[:len(S_init)] = S_init
    spectrum = np.flip(full_S)
    Vh_sorted = np.flip(Vh, axis=0)

    potential_indices = []
    for i in range(gauge_n, num_vars):
        if spectrum[i] < 0.1: potential_indices.append(i)

    # --- Phase 2 ---
    potential_basis_vectors = []
    if len(potential_indices) > 0:
        potential_basis_vectors = Vh_sorted[potential_indices, :]

    idof_vectors = []
    if len(potential_basis_vectors) > 0:
        idof_vectors = detect_instantaneous_modes(
            build_K_matrix, potential_basis_vectors, loops, edge_to_col, node_screw_map
        )

    # --- Phase 3 ---
    if len(idof_vectors) > 0:
        print(f"🔄 [Phase 3] 剔除 {len(idof_vectors)} 个 IDOF...")
        K_final = augment_k_matrix_to_remove_modes(K_initial, idof_vectors, weight=10.0)
    else:
        print("✅ [Phase 3] 未检测到瞬时自由度。")
        K_final = K_initial

    U_f, S_f_raw, Vh_f = np.linalg.svd(K_final, full_matrices=True)
    S_padded = np.zeros(num_vars)
    S_padded[:min(K_final.shape)] = S_f_raw
    final_spectrum = np.flip(S_padded)
    Vh_final_sorted = np.flip(Vh_f, axis=0)
    evecs = Vh_final_sorted.T

    # --- DOF 判定 ---
    valid_evals = final_spectrum[gauge_n:]
    physical_dof = 0
    max_gap = 0.0
    potential_dof_idx = 0
    STRICT_DOF_THRESHOLD = dof_threshold

    if len(valid_evals) > 0:
        for i in range(min(6, len(valid_evals) - 1)):
            v_curr = valid_evals[i] if valid_evals[i] > 1e-12 else 1e-12
            v_next = valid_evals[i + 1]
            gap = v_next / v_curr
            if v_curr < STRICT_DOF_THRESHOLD and gap > 10.0:
                if gap > max_gap:
                    max_gap = gap
                    potential_dof_idx = i + 1

    if max_gap > 10.0:
        physical_dof = potential_dof_idx
    else:
        physical_dof = np.sum(valid_evals < STRICT_DOF_THRESHOLD)

    # --- 详细速度提取 ---
    dof_details = []
    if physical_dof > 0:
        indices = np.arange(gauge_n, gauge_n + int(physical_dof))
        if indices.max() < evecs.shape[1]:
            dof_basis = evecs[:, indices]
            for k in range(dof_basis.shape[1]):
                mode_vec = dof_basis[:, k]
                joint_vels = []
                for edge_idx, vel in enumerate(mode_vec):
                    u, v = directed_edges[edge_idx]
                    joint_vels.append({
                        "edge": (u, v),
                        "vel": float(vel)
                    })
                dof_details.append({
                    "mode_id": k + 1,
                    "velocities": joint_vels
                })

    # --- EE Analysis ---
    null_space_basis = None
    if physical_dof > 0:
        indices = np.arange(gauge_n, gauge_n + int(physical_dof))
        if indices.max() < evecs.shape[1]:
            null_space_basis = evecs[:, indices]

    ee_rank = 0
    motion_desc = "Locked"
    ee_basis_normalized = []

    if null_space_basis is not None:
        # 如果前面 construct_smart_path 成功，extended_path 已经有了
        # 如果没有，尝试旧逻辑
        if extended_path is None:
            if nx.has_path(G_raw, base_node, ee_node):
                raw = nx.shortest_path(G_raw, base_node, ee_node)
                extended_path = _build_extended_path_nx(G_raw, raw)
            else:
                extended_path = []

        J_path = np.zeros((6, num_vars))
        if extended_path and len(extended_path) >= 3:
            for i in range(1, len(extended_path) - 1):
                curr, next_n, prev_n = extended_path[i], extended_path[i + 1], extended_path[i - 1]
                # 这里 prev_n 可能是 None (Ghost)，如果是 None，需要跳过对应的 J 填充吗？
                # 实际上如果 ghost 是 None，意味着它没有前驱，这通常只发生在纯开链的起点头部
                # 但在这里我们只关注 path 的中间部分
                if curr in node_screw_map:
                    screw = node_screw_map[curr]
                    if next_n is not None and (curr, next_n) in edge_to_col:
                        J_path[:, edge_to_col[(curr, next_n)]] += screw
                    if prev_n is not None and (curr, prev_n) in edge_to_col:
                        J_path[:, edge_to_col[(curr, prev_n)]] -= screw

        T_raw = J_path @ null_space_basis
        try:
            U_ee, S_ee, Vh_ee = np.linalg.svd(T_raw, full_matrices=False)
            max_s = S_ee[0] if len(S_ee) > 0 else 0
            ee_rank = np.sum(S_ee > max(1e-6, max_s * 1e-4))
            if ee_rank > 0:
                basis_cols = U_ee[:, :ee_rank]
                ee_basis_normalized = basis_cols.T.tolist()
                if ee_rank == 1:
                    w = basis_cols[:3, 0]
                    v = basis_cols[3:, 0]
                    w_norm = np.linalg.norm(w)
                    v_norm = np.linalg.norm(v)
                    if w_norm < max(1e-5, 1e-4 * max(v_norm, 1e-12)):
                        motion_desc = "1P (Pure Translation)"
                    else:
                        pitch = np.dot(w, v) / (w_norm ** 2)
                        if abs(pitch) < 1e-2:
                            motion_desc = "1R (Pure Rotation)"
                        else:
                            motion_desc = f"1H (Screw, h={pitch:.2f})"
                else:
                    motion_desc = f"{ee_rank}-DOF Spatial"
        except:
            pass

    return {
        "dof": int(physical_dof),
        "idof_count": len(idof_vectors),
        "motion_type": motion_desc,
        "ee_rank": int(ee_rank),
        "connectivity": f"Nodes:{num_nodes}, Edges:{num_vars}, Loops:{num_loops}",
        "ee_twist_basis": ee_basis_normalized,
        "spectrum": final_spectrum.tolist(),
        "gauge_dof": int(gauge_n),
        "dof_details": dof_details
    }
