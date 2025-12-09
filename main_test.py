# main_test.py
import numpy as np
from dof_analysis import analyze_mobility_anchor
from mechanism_utils import load_mechanism_from_json

JSON_FILE_NAME = input("请输入文件名 (默认 Tian_1T1R): ") or "Tian_1T1R"


def run_test():
    print(f"📂 正在加载机构配置: {JSON_FILE_NAME}.json ...")
    try:
        screws, links, base, ee, path, nodes_info, rigid_bodies = load_mechanism_from_json(JSON_FILE_NAME)
        print("✅ 数据加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"🚀 开始分析... (Base: {base} -> EE: {ee})")
    result = analyze_mobility_anchor(
        node_screw_map=screws,
        topology_edges=links,
        nodes_info=nodes_info,
        rigid_body_sets=rigid_bodies,
        base_node=base,
        ee_node=ee,
        manual_extended_path=path
    )
    print_results(result)


def print_results(result):
    if "error" in result:
        print(f"\n❌ 分析中断: {result['error']}")
        return

    print("\n" + "=" * 70)
    print(f"📊 分析报告: {JSON_FILE_NAME}")
    print("=" * 70)

    idof_c = result.get('idof_count', 0)
    if idof_c > 0:
        print(f"⚠️  检测并剔除了 {idof_c} 个瞬时自由度 (IDOF)")
    else:
        print(f"✅ 未检测到瞬时自由度 (纯净机构)")

    if 'spectrum' in result and 'gauge_dof' in result:
        gauge_n = result['gauge_dof']
        raw_spec = result['spectrum']
        auto_dof = result['dof']

        print(f"\n📉 特征值谱 (剔除 {gauge_n} 个规范模态):")
        print(f"   {'Index':<6} | {'SingularVal':<12} | {'Gap (Next/Curr)':<22} | {'Type'}")
        print("-" * 70)

        if len(raw_spec) > gauge_n:
            valid_spec = raw_spec[gauge_n:]
            show_count = min(len(valid_spec), max(10, auto_dof + 3))

            for i in range(show_count):
                val = valid_spec[i]
                ratio_str = "-"
                if i < len(valid_spec) - 1:
                    next_val = valid_spec[i + 1]
                    safe_val = val if val > 1e-12 else 1e-12
                    ratio = next_val / safe_val
                    if ratio > 50.0:
                        ratio_str = f"{ratio:.1e}x 🔥"
                    else:
                        ratio_str = f"{ratio:.1f}x"
                    if i == auto_dof - 1: ratio_str += " (👈 MAX GAP)"

                mark = "✅ DOF" if i < auto_dof else "⛔ Const"
                val_str = f"{val:.2e}" if val < 0.01 else f"{val:.4f}"
                print(f"   {i + 1:<6} | {val_str:<12} | {ratio_str:<22} | {mark}")

                if i == auto_dof - 1: print(f"   {'-' * 66}")
        else:
            print("   (数据不足，无法显示谱分析)")

    print("-" * 70)
    print(f"🔗 拓扑信息:       {result['connectivity']}")
    print(f"⚙️  自动判定DOF:    {result['dof']} (基于 SVD 间隙)")
    print(f"🎯 末端秩 (Rank):  {result['ee_rank']}")
    print(f"📝 运动类型:       {result['motion_type']}")

    print("-" * 70)
    print("🌊 末端螺旋基 (Twist Basis):")
    if result['ee_twist_basis']:
        for i, twist in enumerate(result['ee_twist_basis']):
            fmt_twist = "[ " + ", ".join([f"{x:>8.4f}" for x in twist]) + " ]"
            print(f"  Mode {i + 1}: {fmt_twist}")
    else:
        print("  (Locked / 无有效运动)")

    # =========================================================
    # [新增] 打印详细的关节速度，辅助调试
    # =========================================================
    is_print_velocity = False
    if is_print_velocity:
        dof_details = result.get('dof_details', [])
        if dof_details:
            print("\n" + "=" * 70)
            print("🔍 自由度详细分布 (Joint Velocities Debugger)")
            print("   说明: 显示绝对速度 > 1e-4 的关节。验证方法: sum(Screw * Vel) = 0")
            print("=" * 70)

            for detail in dof_details:
                mode_id = detail['mode_id']
                print(f"\n[Mode {mode_id}] 关节速度分量 (Normalized):")
                print(f"   {'Link Edge (From->To)':<25} | {'Velocity':<12} | {'Bar Graph'}")
                print("-" * 65)

                # 排序：按速度绝对值从大到小
                sorted_vels = sorted(detail['velocities'], key=lambda x: abs(x['vel']), reverse=True)

                has_motion = False
                for item in sorted_vels:
                    edge = item['edge']
                    v = item['vel']

                    # 只显示有明显运动的关节，减少噪声
                    if abs(v) > 1e-4:
                        has_motion = True
                        bar_len = int(abs(v) * 20)  # 简易进度条
                        bar = "█" * bar_len
                        edge_str = f"{edge[0]} -> {edge[1]}"
                        print(f"   {edge_str:<25} | {v:>.4f}      | {bar}")

                if not has_motion:
                    print("   (所有关节速度均接近 0，可能是数值噪声)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_test()