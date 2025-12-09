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


def augment_k_matrix_to_remove_modes(K, bad_modes, weight=10.0):
    if not bad_modes: return K
    rows_to_add = []
    for mode in bad_modes:
        mode_norm = mode / (np.linalg.norm(mode) + 1e-12)
        rows_to_add.append(mode_norm * weight)
    return np.vstack((K, np.array(rows_to_add)))


def detect_instantaneous_modes(K_func_builder, candidate_modes, loops, edge_to_col, node_screw_map):
    """
    使用 [李括号漂移-投影相容性测试] 剔除瞬时自由度 (IDOF)。
    原理：检查二阶几何漂移 (Drift) 是否落在当前雅可比矩阵 (K) 的列空间内。
    """
    # 步长：足够小以满足李代数线性近似
    dt = 1e-3
    idof_vectors = []
    print(f"   🕵️  正在进行多闭环漂移投影检测 (Multi-loop Drift Projection, Step={dt})...")

    # --- 0. 预计算当前构型的“支付能力” (K矩阵) ---
    # 优化：K矩阵只取决于当前几何位置，与尝试哪个 mode 无关，所以提取到循环外
    K_curr = K_func_builder(node_screw_map)

    # 计算伪逆 (Pseudo-inverse)，用于投影
    # rcond=1e-3 用于忽略极小的数值噪声，视情况可微调
    K_pinv = np.linalg.pinv(K_curr, rcond=1e-3)

    for i, mode_vec in enumerate(candidate_modes):
        # 归一化模式向量 (单位速度)
        mode_vec = mode_vec / (np.linalg.norm(mode_vec) + 1e-9)

        # 存储每个闭环的漂移向量，最后拼接
        loop_drifts_list = []

        # --- A. 逐个闭环计算漂移向量 (The "Bill") ---
        for loop in loops:
            loop_drift = np.zeros(6)
            L = len(loop)

            # 用于累积当前环内的几何位置 (杆长/力臂)
            current_twist_sum = np.zeros(6)

            for j in range(L):
                curr_node = loop[j]
                prev_node = loop[(j - 1 + L) % L]
                next_node = loop[(j + 1) % L]

                # 1. 提取关节角速度 (从 mode_vec 映射到 edge)
                val_next = mode_vec[edge_to_col.get((curr_node, next_node), -1)] if (curr_node,
                                                                                     next_node) in edge_to_col else 0.0
                val_prev = mode_vec[edge_to_col.get((curr_node, prev_node), -1)] if (curr_node,
                                                                                     prev_node) in edge_to_col else 0.0

                # 关节相对速度 * 时间步长 = 关节转角增量
                # [注意] 必须乘 dt，否则 current_twist_sum 会过大导致线性近似失效
                d_theta = (val_next - val_prev) * dt

                # 2. 获取当前节点的螺旋 (无需 copy，直接读原始数据)
                screw = node_screw_map[curr_node]

                # 3. 计算李括号 (二阶漂移项)
                # 物理含义: 当前累积的杆长(twist_sum) x 当前转动(screw) -> 产生的额外离心位移
                drift_contribution = _lie_bracket(current_twist_sum, screw)

                # 累加漂移: 漂移速率 * 转角 = 实际漂移量
                loop_drift += drift_contribution * d_theta

                # 4. 更新累积位置 (一阶切线项)
                current_twist_sum += screw * d_theta

            loop_drifts_list.append(loop_drift)

        # --- B. 拼接与投影 (The "Payment") ---
        # 将所有环的漂移拼接成 (6 * NumLoops) 维向量
        full_drift_vector = np.concatenate(loop_drifts_list)

        # 投影测试: 检查 K 矩阵能否产生这个 Drift
        # solution = K_pinv @ drift (尝试凑出账单)
        solution = K_pinv @ full_drift_vector

        # projected = K @ solution (实际能凑出的部分)
        projected_drift = K_curr @ solution

        # --- C. 判据 (The "Verdict") ---
        # 残差 = 想要的 - 实际能给的
        residual_vec = full_drift_vector - projected_drift

        residual_norm = np.linalg.norm(residual_vec)
        drift_norm = np.linalg.norm(full_drift_vector)

        # 防止除零
        if drift_norm < 1e-12:
            ratio = 0.0
        else:
            ratio = residual_norm / drift_norm

        # 阈值判定：如果超过 10% 的漂移无法被补偿，认为是死锁
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
                            manual_extended_path=None,
                            dof_threshold=1e-4):
    # --- 0. 拓扑 ---
    if manual_extended_path is not None:
        if base_node is None: base_node = manual_extended_path[1]
        if ee_node is None: ee_node = manual_extended_path[-2]
    if base_node is None or ee_node is None: return {"error": "Args missing"}

    G_raw = nx.Graph()
    for u, v in topology_edges: G_raw.add_edge(u, v)

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

            # 检查用户定义的刚体
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

    # --- DOF 判定逻辑 (Max Gap Strategy) ---
    valid_evals = final_spectrum[gauge_n:]
    physical_dof = 0
    max_gap = 0.0
    potential_dof_idx = 0

    # [修改建议] 将接受阈值从 0.1 收紧到 1e-4 或 1e-5
    # 只有当奇异值真的非常接近 0 时，我们才关心它后面的 Gap
    STRICT_DOF_THRESHOLD = dof_threshold

    if len(valid_evals) > 0:
        for i in range(min(6, len(valid_evals) - 1)):
            v_curr = valid_evals[i] if valid_evals[i] > 1e-12 else 1e-12
            v_next = valid_evals[i + 1]
            gap = v_next / v_curr

            # 修改这里的判断条件：v_curr < STRICT_DOF_THRESHOLD
            if v_curr < STRICT_DOF_THRESHOLD and gap > 10.0:
                if gap > max_gap:
                    max_gap = gap
                    potential_dof_idx = i + 1

    if max_gap > 10.0:
        physical_dof = potential_dof_idx
    else:
        # 兜底逻辑也使用严格阈值
        physical_dof = np.sum(valid_evals < STRICT_DOF_THRESHOLD)

    # ========================================================
    # [新增] 提取详细的关节速度分布 (Debugging Info)
    # ========================================================
    dof_details = []
    if physical_dof > 0:
        # 提取对应物理自由度的基向量 (跳过 gauge_n)
        indices = np.arange(gauge_n, gauge_n + int(physical_dof))
        if indices.max() < evecs.shape[1]:
            dof_basis = evecs[:, indices]

            # 遍历每一个找到的自由度模式
            for k in range(dof_basis.shape[1]):
                mode_vec = dof_basis[:, k]

                # 记录该模式下所有边的速度
                joint_vels = []
                for edge_idx, vel in enumerate(mode_vec):
                    u, v = directed_edges[edge_idx]
                    # 只记录绝对值大于极小值的，或者全部记录方便排查
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
        if manual_extended_path:
            extended_path = manual_extended_path
        else:
            if nx.has_path(G_raw, base_node, ee_node):
                raw = nx.shortest_path(G_raw, base_node, ee_node)
                extended_path = _build_extended_path_nx(G_raw, raw)
            else:
                extended_path = []

        J_path = np.zeros((6, num_vars))
        if extended_path and len(extended_path) >= 3:
            for i in range(1, len(extended_path) - 1):
                curr, next_n, prev_n = extended_path[i], extended_path[i + 1], extended_path[i - 1]
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
                    if np.linalg.norm(w) < 1e-5:
                        motion_desc = "1P (Pure Translation)"
                    else:
                        pitch = np.dot(w, basis_cols[3:, 0]) / (np.linalg.norm(w) ** 2)
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
        "dof_details": dof_details  # 返回详细速度信息
    }
