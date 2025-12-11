# Generalised Mobility Analysis Framework
# 通用机构自由度分析框架

这是一个基于**螺旋理论 (Screw Theory)** 和 **奇异值分解 (SVD)** 的通用机构自由度分析工具。该框架专为分析过约束机构、空间并联机构以及包含瞬时自由度 (IDOF) 的复杂机构而设计。

它不仅能计算机构的物理自由度 (DOF)，还能通过二阶李代数（Lie Bracket）漂移检测算法识别并剔除瞬时自由度，并分析末端执行器 (End-Effector) 的运动性质。

## 🌟 核心功能 (Key Features)

* **通用自由度计算**: 不依赖于传统的 Grubler-Kutzbach 公式，而是基于雅可比矩阵的秩分析，能够准确处理过约束机构（如 Bennett 机构、Sarrus 机构）。
* **瞬时自由度 (IDOF) 检测与剔除**:
    * 包含二阶分析模块，利用**李括号 (Lie Bracket)** 计算闭环漂移。
    * 通过投影相容性测试（Drift Projection）自动识别并剔除奇异位形下的瞬时自由度。
* **特征值谱分析 (Spectrum Analysis)**:
    * 使用 SVD 分析雅可比矩阵的奇异值。
    * 通过 "Max Gap" 策略自动判定有效自由度数量。
    * 自动处理刚体变换带来的规范模态 (Gauge Modes)。
* **末端运动分析**:
    * 计算末端执行器的运动空间维数 (Rank)。
    * 识别末端运动类型（如：1T1R, 纯移动, 纯转动, 空间运动等）。
* **单位量纲自动平衡**: 自动计算机构特征长度 ($L_{char}$)，用于平衡转动副 (R) 和移动副 (P) 在数值计算中的量纲差异。

## 📂 项目结构 (File Structure)

```text
.
├── dof_analysis.py         # [核心] 自由度分析算法引擎 (SVD, IDOF检测, 运动空间分析)
├── mechanism_utils.py      # [工具] JSON配置加载、螺旋坐标转换、特征长度计算
├── main_test.py            # [入口] 主测试脚本，包含结果打印和可视化报告
├── mechanisms/             # [数据] 存放机构定义的 JSON 文件
│   ├── Tian_1T1R.json      # 示例：Tian 1T1R 并联机构
│   ├── bennett.json        # 示例：Bennett 空间过约束连杆
│   ├── planar_3RRR.json    # 示例：平面 3RRR 机构
│   ├── spatial_3RRR.json   # 示例：空间 3RRR 机构
│   └── ...
└── README.md               # 说明文档
```

## 🚀 快速开始 (Quick Start)

### 环境要求
本项目依赖 Python 3.x 及以下科学计算库：
```bash
pip install numpy networkx
```

### 运行分析
在终端中运行 `main_test.py`：

```bash
python main_test.py
```

程序启动后，会提示输入要分析的 JSON 文件名（无需后缀）：
```text
请输入文件名 (默认 Tian_1T1R): bennett
```

### 输出示例
程序将输出详细的分析报告，包括：
1.  **IDOF 剔除情况**: 是否检测到瞬时自由度。
2.  **特征值谱**: 雅可比矩阵的奇异值分布及间隙比率 (Gap Ratio)。
3.  **自由度判定**: 基于 SVD 间隙得出的物理自由度数。
4.  **末端分析**: 末端的运动秩及螺旋基向量。

```text
📊 分析报告: bennett
======================================================================
✅ 未检测到瞬时自由度 (纯净机构)

📉 特征值谱 (剔除 4 个规范模态):
   Index  | SingularVal  | Gap (Next/Curr)        | Type
----------------------------------------------------------------------
   1      | 2.65e+00     | 1.0x                   | ✅ DOF
   2      | 1.00e-15     | 2.6e+15x 🔥 (👈 MAX GAP)| ⛔ Const
----------------------------------------------------------------------
🔗 拓扑信息:       Nodes:4, Edges:4, Loops:1
⚙️  自动判定DOF:    1 (基于 SVD 间隙)
🎯 末端秩 (Rank):  1
📝 运动类型:       1H (Screw, h=0.15)
...
```

## 🛠 如何自定义机构 (Custom Mechanism)

在 `mechanisms/` 目录下创建一个新的 `.json` 文件。文件格式规范如下：

### JSON 结构说明

```json
{
  "name": "My Mechanism Name",
  "nodes": [
    {
      "id": 0,                // 节点唯一ID (整数)
      "type": "R",            // 关节类型: "R" (转动副) 或 "P" (移动副)
      "axis": [0, 0, 1],      // 关节轴线向量 [x, y, z]
      "pos": [0, 0, 0]        // 关节位置坐标 [x, y, z]
    },
    ...
  ],
  "links": [
    [0, 1],                   // 定义连杆连接关系 [节点ID_A, 节点ID_B]
    [1, 2],
    ...
  ],
  "rigid_bodies": [           // (可选) 定义刚性耦合体，通常用于表示三角形构件
    [0, 1, 2]                 // 这些节点之间视为刚性连接，无相对运动
  ],
  "settings": {
    "manual_path": [0, 1, 2], // (可选) 指定一条从基座到末端的路径链，用于末端运动分析
    "base_node": 0,           // 基座节点 ID
    "ee_node": 2              // 末端节点 ID
  }
}
```

### 关键字段详解
1.  **`nodes`**: 定义所有运动副的位置和方向。`mechanism_utils.py` 会自动将其转换为 Plücker 坐标（螺旋）。
2.  **`links`**: 定义机构的拓扑图。
3.  **`settings` -> `manual_path`**:
    * 如果您关心末端执行器（End-Effector）的性质，**必须**提供此路径。
    * 格式为节点 ID 的有序列表，代表从 Base 到 EE 的一条运动链。
    * 如果未提供，程序仅能计算整体自由度，无法给出具体的末端运动类型。

## 🧠 算法原理 (Algorithm Overview)

1.  **构建运动旋量系统**: 将所有关节转换为螺旋 $\mathbf{S}_i$。对于 R 副，$\mathbf{S} = [\boldsymbol{\omega}; \boldsymbol{r} \times \boldsymbol{\omega}]$；对于 P 副，$\mathbf{S} = [\mathbf{0}; \mathbf{v}]$。
2.  **雅可比矩阵构建**: 基于闭环约束方程 $\sum \dot{\theta}_i \mathbf{S}_i = \mathbf{0}$ 构建全局雅可比矩阵 $\mathbf{K}$。
3.  **IDOF 检测 (Phase 2)**:
    * 计算二阶李括号项 $[\mathbf{S}_i, \mathbf{S}_j]$。
    * 如果某模式的一阶运动导致的二阶几何漂移（Drift）无法被机构现有的雅可比列空间所覆盖（投影残差大），则判定该模式为瞬时自由度并将其剔除。
4.  **奇异值分析 (Phase 3)**:
    * 对修正后的矩阵进行 SVD。
    * 寻找奇异值谱中的最大间隙（Max Gap），间隙之前的大奇异值数量即为机构的有效自由度。
