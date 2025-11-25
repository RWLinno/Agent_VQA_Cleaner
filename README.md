# AgentCleaner - VQA数据智能清洗与点位验证系统

基于多模态大模型（VLM）的视觉问答（VQA）数据质量评估与清洗系统，支持通用VQA数据清洗和点定位任务的专项验证。

## ✨ 主要特性

### 🎯 核心功能

#### 1. VQA数据质量评估与清洗
- **多模态质量评估**: 使用 VLM 评估图像质量、问题合理性和答案准确性
- **智能路径查找**: 自动在多个子目录中查找图像和深度图
- **3D数据支持**: 完整支持 RGB + 深度图的多模态数据处理
- **启发式规则**: 自动检测重复模式、超长句子、尾部循环等问题
- **点数据检测**: 自动识别并处理答案中的坐标点数据

#### 2. 点定位任务验证（专项功能）
- **点位精度验证**: 使用VLM重新推理验证标注点位的准确性
- **负样本交叉验证**: 通过扰动生成负样本，验证模型判断的可靠性
- **三重验证机制**: 点位匹配 + PointQA正确性 + 负样本识别
- **详细失败分析**: 记录每个样本的失败原因，便于数据质量分析
- **可视化对比**: 直观展示原始点、推理点、扰动点的位置关系

#### 3. 系统特性
- **并发处理**: 多处理器并行处理，充分利用GPU资源
- **增量保存**: 自动保存中间结果，防止数据丢失
- **双模式支持**: 支持本地模型和阿里云EAS API两种模式

### 🚀 性能优势
- **默认模型**: Qwen2.5-VL-3B-Instruct（仅3B参数，显存~8GB）
- **云端部署**: 支持阿里云EAS API（如Qwen3-VL-235B），无需本地GPU
- **处理速度**: 
  - VQA清洗: ~100-150 样本/分钟（4 GPUs, 8 processors）
  - 点位验证: ~60-90 样本/分钟（包含3次推理/样本）
- **灵活配置**: 支持命令行、环境变量、配置文件多种配置方式
- **完整可视化**: HTML报告直观展示验证结果和点位对比

### 📊 支持的数据格式

#### VQA数据清洗

| 数据集 | 类型 | 图像 | 深度图 | 支持状态 |
|--------|------|------|--------|---------|
| RefSpatial 2D | choice_qa | 单图 | ❌ | ✅ |
| RefSpatial 2D | reasoning_template_qa | 单图 | ❌ | ✅ |
| RefSpatial 3D | choice_qa | 单图 | ✅ | ✅ |
| RefSpatial 3D | multi_view_qa | 多图 | ✅ | ✅ |
| RefSpatial 3D | visual_choice_qa | 单图+bbox | ✅ | ✅ |
| RefSpatial 3D | reasoning_template_qa | 单图 | ✅ | ✅ |
| RefSpatial 3D | vacant_qa | 单图 | ✅ | ✅ |
| 通用VQA | 自定义格式 | 单图/多图 | 可选 | ✅ |

#### 点位验证（专项）

| 任务类型 | 数据格式 | 验证方式 | 支持状态 |
|---------|---------|---------|---------|
| 点定位标注 | JSONL (messages格式) | 点位对比 + PointQA | ✅ |
| 目标检测 | 带坐标答案 | 三重验证 | ✅ |
| 关键点检测 | 多点标注 | 负样本交叉验证 | ✅ |

## 🚀 快速开始

### 功能选择

本系统提供两大功能模块：

1. **VQA数据清洗** (`main.py`) - 通用VQA数据质量评估
2. **点位验证** (`main_point_verification.py`) - 点定位任务的专项验证

根据您的需求选择对应的功能。

---

### 1. 环境准备

```bash
# 克隆仓库
git clone <repository_url>
cd Agent_VQA_Cleaner

# 安装依赖
pip install -r requirements.txt

# 安装最新版transformers（支持Qwen2.5-VL）
pip install git+https://github.com/huggingface/transformers.git accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

### 2. 配置GPU

```bash
# 设置可用GPU
export CUDA_VISIBLE_DEVICES=0,1

# 设置并发参数
export NUM_PROCESSORS=8
export BATCH_SIZE=4
```

### 3. 运行功能

#### 功能A: VQA数据清洗

##### 方式1: 2D数据清洗
```bash
# 编辑 quick_start.sh 中的路径配置
vim quick_start.sh

# 运行清洗
./quick_start.sh
```

##### 方式2: 3D数据清洗（单个数据集）
```bash
# 编辑 quick_start_3d.sh 中的路径配置
vim quick_start_3d.sh

# 运行清洗
./quick_start_3d.sh
```

##### 方式3: 批量处理所有3D数据（推荐）⭐
```bash
# 编辑 batch_clean_3d.sh 中的BASE_3D路径
vim batch_clean_3d.sh

# 批量运行
./batch_clean_3d.sh
```

##### 方式4: 使用阿里云EAS API（无需本地GPU）
```bash
# 编辑 quick_start_eas.sh 配置EAS服务信息
vim quick_start_eas.sh

# 运行清洗（使用云端大模型）
./quick_start_eas.sh
```

**特点**：
- ✅ 无需本地GPU资源
- ✅ 支持超大规模模型（如235B参数）
- ✅ 配置简单，只需API地址和token

#### 功能B: 点位验证（点定位任务专用）

用于验证点定位标注的准确性，包含负样本交叉验证。

##### 快速测试（10个样本）

```bash
# 1. 配置EAS信息
export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url"
export EAS_TOKEN="your-token"

# 2. 运行测试
./test_perturbation_verification.sh

# 3. 查看结果
firefox output/test_perturbation/visualization.html
```

##### 完整数据处理

```bash
# 基本用法
python main_point_verification.py \
    --input /path/to/data.jsonl \
    --root_path /path/to/images \
    --output output/verification_results.jsonl \
    --num_workers 8

# 启用所有验证功能（推荐）
python main_point_verification.py \
    --input /path/to/data.jsonl \
    --root_path /path/to/images \
    --output output/verification_results.jsonl \
    --num_workers 8 \
    --distance_threshold 50 \
    --enable_perturbation true \
    --perturbation_range 100

# 生成可视化报告
python visualize_point_verification.py \
    --input output/verification_results.jsonl \
    --output output/visualization.html \
    --filter all \
    --sample_size 50
```

**验证机制说明**：

1. **点位对比**：VLM重新推理生成点P2，与原标注点P1对比距离
2. **PointQA验证**：让模型判断原标注点P1是否正确
3. **负样本验证**：生成扰动点P1'（故意错误），验证模型能否识别错误
4. **三重筛选**：只有同时通过以上三项验证的数据才保留

**关键指标**：
- **点位匹配率**：P1和P2距离<阈值的比例（期望60-80%）
- **PointQA正确率**：P1被判定正确的比例（期望50-70%）
- **负样本检测率**：能正确识别P1'错误的比例（期望>70%，关键指标）
- **严格筛选通过率**：三重验证都通过的比例（期望30-50%）

### 4. 查看结果

#### VQA清洗结果

清洗完成后，会生成以下文件：
- `*_cleaned.jsonl`: 清洗后的高质量数据（用于训练）
- `*_full.jsonl`: 完整评估结果（包含所有元数据）
- `*_system_prompt.txt`: 使用的评估提示词

#### 点位验证结果

验证完成后，会生成：
- `*_results.jsonl`: 完整验证结果（包含点位对比、PointQA验证、负样本检测）
- `*_verification.log`: 详细日志
- `visualization.html`: 可视化报告（显示P1/P2/P1'三个点的位置）

**可视化说明**：
- 🔴 **P1（红色）**：原始标注点（正样本）
- 🟢 **P2（绿色）**：VLM推理生成的点
- 🟠 **P1'（橙色）**：扰动点（负样本，用于验证可靠性）
- 黄色虚线：P1↔P2（距离越短越好）
- 橙色虚线：P1↔P1'（扰动距离）

## 📂 数据配置指南

### 2D 数据配置

编辑 `quick_start.sh`：
```bash
# 配置路径
JSON_FILE="/path/to/your/2D/choice_qa.json"
ROOT_PATH="/path/to/your/2D/image/"
OUTPUT_FILE="./output/2D/choice_qa_cleaned.jsonl"
```

**目录结构**：
```
2D/
├── image/
│   └── xxx.jpg          ← 所有图像在这里
└── choice_qa.json       ← JSON包含相对路径
```

### 3D 数据配置 ⭐

编辑 `batch_clean_3d.sh`：
```bash
# 配置3D数据根目录
BASE_3D="/path/to/your/3D"
ROOT_PATH="$BASE_3D/"  # 重要：指向根目录！
```

**目录结构**：
```
3D/
├── image/              ← RGB图像
│   └── xxx_image.png
├── depth/              ← 深度图
│   └── xxx_depth.png
├── image_multi_view/   ← 多视角RGB
├── depth_multi_view/   ← 多视角深度
├── image_visual_choice/ ← 带bbox的图像
├── choice_qa.json
├── multi_view_qa.json
├── visual_choice_qa.json
├── reasoning_template_qa.json
└── vacant_qa.json
```

**关键说明**：
- JSON 中只存储文件名（如 `xxx_image.png`），不包含子目录路径
- `ROOT_PATH` 应指向 `3D/` 根目录
- 系统会自动在 `image/`、`depth/` 等子目录中智能查找文件

## 🎯 命令行使用

### VQA数据清洗 (main.py)

#### 基本用法

```bash
python main.py \
    --json_file /path/to/data.json \
    --root_path /path/to/images/ \
    --output ./output/cleaned.jsonl
```

### 处理指定范围

```bash
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --start 0 \
    --end 1000  # 只处理前1000条
```

### 包含答案评估

```bash
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --include_answer  # 评估时包含答案
```

### 自定义配置

```bash
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --num_processors 16 \
    --batch_size 8 \
    --score_threshold 6
```

#### 禁用启发式规则

```bash
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --no_heuristic  # 只使用VLM评分
```

---

### 点位验证 (main_point_verification.py)

#### 基本用法

```bash
python main_point_verification.py \
    --input /path/to/data.jsonl \
    --root_path /path/to/images/ \
    --output ./output/verification_results.jsonl
```

#### 启用负样本验证（推荐）

```bash
python main_point_verification.py \
    --input data.jsonl \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --enable_perturbation true \
    --perturbation_range 100
```

#### 调整验证参数

```bash
python main_point_verification.py \
    --input data.jsonl \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --distance_threshold 50 \
    --enable_perturbation true \
    --perturbation_range 100 \
    --num_workers 8
```

#### 禁用负样本验证（快速模式）

```bash
python main_point_verification.py \
    --input data.jsonl \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --enable_perturbation false  # 禁用扰动验证
```

#### 处理指定范围

```bash
# 只处理前100条数据
python main_point_verification.py \
    --input data.jsonl \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --start 0 \
    --end 100
```

---

### 可视化工具 (visualize_point_verification.py)

#### 生成可视化报告

```bash
# 显示所有样本
python visualize_point_verification.py \
    --input verification_results.jsonl \
    --output visualization_all.html \
    --filter all

# 只显示通过严格筛选的样本
python visualize_point_verification.py \
    --input verification_results.jsonl \
    --output visualization_pass.html \
    --filter strict_pass

# 只显示失败的样本（重点分析）
python visualize_point_verification.py \
    --input verification_results.jsonl \
    --output visualization_fail.html \
    --filter strict_fail

# 只显示负样本检测失败的（验证不可靠的）
python visualize_point_verification.py \
    --input verification_results.jsonl \
    --output visualization_neg_fail.html \
    --filter negative_failed
```

#### 过滤模式说明

| 模式 | 说明 | 用途 |
|------|------|------|
| `all` | 显示所有样本 | 整体浏览 |
| `matched` | 点位匹配的样本 | 查看P1和P2接近的 |
| `mismatched` | 点位不匹配的样本 | 查看P1和P2距离远的 |
| `correct` | PointQA判定正确的 | 查看被认为准确的标注 |
| `incorrect` | PointQA判定不正确的 | 查看被认为错误的标注 |
| `strict_pass` | 通过严格筛选的 | 高质量数据 |
| `strict_fail` | 未通过严格筛选的 | 需要重点分析 |
| `negative_failed` | 负样本检测失败的 | 验证不可靠的样本 |

## ⚙️ 配置参数

### VQA清洗参数 (main.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--json_file` | 输入JSON文件路径 | 必需 |
| `--root_path` | 图像根目录路径 | 必需 |
| `--output` | 输出文件路径 | 必需 |
| `--start` | 起始索引 | 0 |
| `--end` | 结束索引 | None |
| `--num_processors` | 并发处理器数量 | 8 |
| `--batch_size` | 批处理大小 | 4 |
| `--score_threshold` | 评分阈值 | 5 |
| `--include_answer` | 评估时包含答案 | False |
| `--no_concurrent` | 禁用并发处理 | False |
| `--no_heuristic` | 禁用启发式规则 | False |
| `--log_level` | 日志级别 | INFO |
| `--log_file` | 日志文件路径 | None |

### 点位验证参数 (main_point_verification.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入JSONL文件路径 | 必需 |
| `--root_path` | 图像根目录路径 | 必需 |
| `--output` | 输出JSONL文件路径 | 必需 |
| `--num_workers` | 并行worker数量 | 4 |
| `--distance_threshold` | 点位距离阈值（像素） | 50.0 |
| `--enable_perturbation` | 启用点位扰动验证 | True |
| `--perturbation_range` | 扰动范围（像素） | 100 |
| `--start` | 起始索引 | 0 |
| `--end` | 结束索引 | None |
| `--log_level` | 日志级别 | INFO |
| `--log_file` | 日志文件路径 | None |

**参数说明**：

- `distance_threshold`: P1和P2的最大允许距离，超过视为不匹配
- `enable_perturbation`: 是否启用负样本验证（推荐开启）
- `perturbation_range`: 扰动点与原点的距离范围

### 可视化参数 (visualize_point_verification.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入JSONL文件路径 | 必需 |
| `--output` | 输出HTML文件路径 | 必需 |
| `--sample_size` | 抽样数量 | 50 |
| `--filter` | 过滤模式 | all |
| `--seed` | 随机种子 | 42 |

### 环境变量

```bash
# GPU设置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 并发配置
export NUM_PROCESSORS=8
export BATCH_SIZE=4

# 模型路径（可选）
export PRIMARY_MODEL=/path/to/model
```

### 性能调优

| 数据类型 | 推荐 BATCH_SIZE | 推荐 PROCESSORS | 说明 |
|----------|-----------------|-----------------|------|
| 2D单图 | 4-8 | 8-16 | 标准配置 |
| 3D单图+深度 | 4-6 | 8-12 | 多图输入 |
| 3D多视角 | 2-4 | 4-8 | 内存密集 |

**调优建议**：
- GPU内存不足 → 减小 `BATCH_SIZE`
- 处理速度慢 → 增加 `NUM_PROCESSORS`
- 多视角数据 → 自动调整（`batch_clean_3d.sh` 已优化）

## 📊 输出文件说明

### 清洗后数据（`*_cleaned.jsonl`）
只包含通过筛选的高质量数据，适合模型训练：
```jsonl
{"question": "问题文本", "answer": "答案文本", "image_path": "xxx.jpg", "score": 8}
```

### 完整数据（`*_full.jsonl`）
包含所有评估信息，适合结果分析：
```jsonl
{
  "question": "问题文本",
  "answer": "答案文本",
  "image_path": "xxx_image.png",
  "depth_path": "xxx_depth.png",
  "full_image_path": "/full/path/to/image/xxx_image.png",
  "full_depth_path": "/full/path/to/depth/xxx_depth.png",
  "score": 8,
  "raw_response": "8",
  "filtered": false,
  "filter_reason": null,
  "processor_id": 2
}
```

### 系统提示词（`*_system_prompt.txt`）
使用的评估提示词，用于复现实验。

## 🏗️ 项目结构

```
Agent_VQA_Cleaner/
├── main.py                              # VQA数据清洗主程序
├── main_point_verification.py           # 点位验证主程序
├── visualize_samples.py                 # VQA清洗结果可视化
├── visualize_point_verification.py      # 点位验证结果可视化
├── requirements.txt                     # 依赖列表
├── README.md                            # 本文档
│
├── quick_start.sh                       # 2D数据清洗快速启动
├── quick_start_3d.sh                    # 3D数据清洗快速启动
├── quick_start_eas.sh                   # EAS API清洗快速启动
├── batch_clean_3d.sh                    # 批量处理3D数据
├── test_perturbation_verification.sh    # 点位验证快速测试
│
├── src/                                 # 源代码
│   ├── config.py                        # 配置管理
│   ├── base/                            # 核心组件
│   │   ├── vlm_agent.py                 # VLM Agent（本地模型）
│   │   ├── vlm_agent_eas.py             # VLM Agent（EAS API）
│   │   ├── processor.py                 # VQA数据处理器
│   │   ├── point_verification_processor.py  # 点位验证处理器
│   │   └── unified_manager.py           # 统一管理器
│   └── utils/                           # 工具函数
│       ├── Data_Selector.py             # 数据选择器
│       └── Heuristic_Rules_Internvl2_5.py  # 启发式规则
│
├── scripts/                             # 辅助脚本
│   ├── extract_fields.py                # 字段提取工具
│   ├── batch_process.sh                 # 多GPU批处理
│   └── analyze_results.py               # 结果分析
│
├── output/                              # 输出目录
│   ├── 2D/                              # 2D清洗输出
│   ├── 3D/                              # 3D清洗输出
│   ├── washing_machine/                 # 洗衣机数据清洗输出
│   └── test_perturbation/               # 点位验证测试输出
│
└── examples/                            # 示例数据
    ├── choice_qa_test.json              # 测试数据
    └── images/                          # 测试图像
```

## 📚 点位验证详解

### 什么是点位验证？

点位验证是针对**点定位标注任务**的专项质量评估功能，用于：
- 验证标注点位的准确性
- 检测标注数据的系统性错误
- 通过负样本交叉验证确保评估可靠性

### 验证流程

```
输入: 标注数据（包含问题、图片、答案中的点坐标）
  ↓
Step 1: 提取原始标注点P1
  ↓
Step 2: VLM重新推理生成点P2
  ↓
Step 3: 计算P1和P2的距离 → 点位匹配
  ↓
Step 4: PointQA验证P1是否正确 → 标注准确性
  ↓
Step 5: 生成扰动点P1'（故意错误）
  ↓
Step 6: PointQA验证P1'是否被识别为错误 → 验证可靠性
  ↓
Step 7: 三重判定
  - P1和P2距离 < 阈值 ✓
  - PointQA判定P1正确 ✓
  - PointQA判定P1'不正确 ✓
  ↓
输出: 通过/失败 + 详细原因
```

### 为什么需要负样本验证？

**问题场景**：如果标注点P1和推理点P2都错了但恰好距离很近怎么办？

**解决方案**：通过扰动生成明显错误的点P1'，验证模型能否正确识别错误
- 如果P1'被判定为不正确 → 模型判断可靠 ✓
- 如果P1'也被判定为正确 → 模型判断不可靠 ✗（过滤该数据）

**关键优势**：
- 避免"两个错误点恰好接近"的误判
- 确保PointQA验证的可靠性
- 提供更严格的数据质量保证

### 输出数据格式

```json
{
  "id": "sample_001",
  "question": "Point to the power button...",
  "label": "power button",
  
  "point_original": {
    "coordinates": [420.91, 263.16]
  },
  
  "point_inferred": {
    "coordinates": [425.0, 265.0],
    "success": true
  },
  
  "point_comparison": {
    "distance": 5.12,
    "match": true,
    "threshold": 50.0
  },
  
  "pointqa_verification": {
    "is_correct": true,
    "confidence": 8.5
  },
  
  "perturbation_verification": {
    "perturbed_point": [520.5, 350.2],
    "distance_from_original": 115.3,
    "is_correct": false,
    "negative_detected": true
  },
  
  "strict_filtering": {
    "pass": true,
    "fail_reasons": []
  }
}
```

---

## 🔍 技术实现

### 智能路径查找

系统自动在多个子目录中查找文件，支持复杂的目录结构：

```python
def _find_image_path(root_path, filename, is_depth=False):
    """智能查找图像文件路径"""
    # 1. 尝试直接路径
    if exists(root_path/filename):
        return root_path/filename
    
    # 2. 在子目录中查找
    if is_depth:
        # 深度图目录: depth/, depth_multi_view/
        ...
    else:
        # RGB图像目录: image/, image_multi_view/, image_visual_choice/
        ...
```

### 多图输入处理

支持 RGB + Depth 的多模态输入：

```python
# 加载图像
images = []
images.append(load_image(image_path))  # RGB
if depth_path:
    images.append(load_image(depth_path))  # Depth

# 传递给VLM模型
image_data = images if len(images) > 1 else images[0]
score, response = vlm_agent.evaluate(question, image_data)
```

### 启发式规则

自动检测常见问题：
- **N-gram重复**: 检测对话中的重复模式
- **超长句子**: 标记异常长的句子
- **尾部循环**: 检测答案末尾的重复
- **循环模式**: 检测长句子+尾部重复的组合

## 🐛 故障排查

### 问题1: 图像文件未找到

```
[ERROR] xxx_image.png | Score: -1 | Reason: Image not found
```

**解决方案**：
- 检查 `ROOT_PATH` 是否正确
- 对于3D数据，确保 `ROOT_PATH` 指向 `3D/` 根目录（不是 `3D/image/`）
- 检查文件是否真实存在

### 问题2: 深度图未找到

```
[ERROR] xxx_depth.png | Score: -1 | Reason: Depth image not found
```

**解决方案**：
- 确认深度图文件存在于 `depth/` 或 `depth_multi_view/` 目录
- 检查 JSON 中的 depth 字段路径
- 使用 `ls` 命令验证文件存在

### 问题3: GPU内存溢出

```
CUDA out of memory
```

**解决方案**：
```bash
# 方案1: 减小 BATCH_SIZE
export BATCH_SIZE=2

# 方案2: 减少 PROCESSORS
export NUM_PROCESSORS=4

# 方案3: 使用单GPU
export CUDA_VISIBLE_DEVICES=0
```

### 问题4: transformers 版本错误

```
ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration'
```

**解决方案**：
```bash
# 安装最新版 transformers
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

### 问题5: 处理速度慢

**优化建议**：
```bash
# 增加并发数
export NUM_PROCESSORS=16

# 增加batch size（如果GPU内存充足）
export BATCH_SIZE=8

# 使用更多GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 📈 性能基准

### 处理速度（4 GPUs, 8 processors）

| 数据类型 | 速度 | 说明 |
|----------|------|------|
| 2D单图 | 100-150 样本/分 | 标准性能 |
| 3D单图+深度 | 80-120 样本/分 | 多图输入 |
| 3D多视角（2图） | 50-80 样本/分 | 内存密集 |
| 3D多视角（4图+） | 30-50 样本/分 | 高度密集 |

### GPU内存占用

| 配置 | 内存占用 | 说明 |
|------|----------|------|
| Qwen2.5-VL-3B (单图) | ~8GB | 每GPU |
| Qwen2.5-VL-3B (多图) | ~12GB | 每GPU |
| Batch=4, Proc=8 | ~32GB | 总占用(4 GPUs) |

## 🎓 使用技巧

### 1. 测试少量数据

```bash
# 先处理100条测试
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output test.jsonl \
    --end 100

# 检查结果
head -10 test.jsonl
```

### 2. 恢复中断的任务

```bash
# 假设处理到第5000条中断了
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --start 5000  # 从5000继续
```

### 3. 调整评分阈值

```bash
# 更严格的筛选（保留高质量）
--score_threshold 7

# 更宽松的筛选（保留更多数据）
--score_threshold 4
```

### 4. 查看详细日志

```bash
# 设置日志级别为DEBUG
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --log_level DEBUG \
    --log_file debug.log

# 查看日志
tail -f debug.log
```

## 📝 数据格式说明

### 输入格式

支持以下JSON格式：

```json
{
  "id": "sample_001",
  "image": ["image.png"],
  "depth": ["depth.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n问题文本"
    },
    {
      "from": "gpt",
      "value": "答案文本"
    }
  ]
}
```

**字段说明**：
- `image`: 图像文件名（字符串或列表）
- `depth`: 深度图文件名（可选，字符串或列表）
- `conversations`: 对话列表，包含 human 和 gpt 的交互

## 📞 常见问题

### 如何选择合适的评分阈值？

默认阈值为5（满分10分）。建议：
- 高质量要求：设置为 6-7
- 平衡质量和数量：使用默认 5
- 保留更多数据：设置为 3-4

### 如何处理超大数据集？

1. 使用 `--start` 和 `--end` 分段处理
2. 利用中间结果保存机制
3. 增加 GPU 和并发数
4. 考虑分布式处理

### 如何自定义评估标准？

编辑 `src/config.py` 中的：
- `SYSTEM_PROMPT_IMAGE_QUESTION`: 图像+问题评估提示
- `SYSTEM_PROMPT_WITH_ANSWER`: 包含答案的评估提示

### 如何集成到训练流程？

清洗后的 `*_cleaned.jsonl` 文件可直接用于训练：

```python
import json

# 读取清洗后的数据
with open('output_cleaned.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        # sample包含: question, answer, image_path, score
        # 进行训练...
```

## 🌐 使用阿里云EAS API

### 快速开始

```bash
# 1. 配置EAS信息
export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url"
export EAS_TOKEN="your-token"
export EAS_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"

# 2. 运行清洗
./quick_start_eas.sh
```

### 测试连接

使用测试脚本验证EAS配置：

```bash
# 测试EAS API连接
python test_eas_connection.py
```

### 详细说明

完整的EAS使用指南请参考：[README_EAS.md](README_EAS.md)

包含：
- EAS服务配置详解
- API调用说明
- 并发配置建议
- 故障排查指南
- 成本控制建议

## 💡 使用建议

### VQA数据清洗

1. **先测试少量数据**（100条）确认配置正确
2. **查看可视化结果**了解过滤原因
3. **调整阈值**平衡质量和数量
4. **批量处理**完整数据集

### 点位验证

1. **快速测试**（10条）验证功能：`./test_perturbation_verification.sh`
2. **查看统计指标**：重点关注负样本检测率（期望>70%）
3. **分析失败样本**：查看`strict_fail`和`negative_failed`的可视化
4. **调整参数**：
   - 负样本检测率低 → 增大`perturbation_range`或检查label定义
   - 通过率过低 → 放宽`distance_threshold`或禁用`enable_perturbation`
5. **全量处理**：确认效果后处理完整数据集

### 参数调优

| 场景 | distance_threshold | perturbation_range | enable_perturbation |
|------|-------------------|-------------------|---------------------|
| 精确定位任务 | 30 | 80 | true |
| 一般定位任务 | 50 | 100 | true |
| 粗略定位任务 | 80 | 120 | true |
| 快速验证模式 | 50 | - | false |

---

## 📄 许可证

本项目遵循 MIT 许可证。

## 🙏 致谢

- Qwen2.5-VL / Qwen3-VL 模型团队
- InternVL 模型团队
- 阿里云PAI-EAS团队
- 所有开源贡献者
