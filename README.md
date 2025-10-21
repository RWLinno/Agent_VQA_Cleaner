# AgentCleaner - VQA数据智能清洗系统

基于多模态大模型（VLM）的视觉问答（VQA）数据质量评估与清洗系统。通过多个开源模型协同工作，自动过滤低质量数据，保留高质量的训练样本。

## ✨ 主要特性

### 🎯 核心功能
- **多模态质量评估**: 使用 VLM 评估图像质量、问题合理性和答案准确性
- **智能路径查找**: 自动在多个子目录中查找图像和深度图
- **3D数据支持**: 完整支持 RGB + 深度图的多模态数据处理
- **启发式规则**: 自动检测重复模式、超长句子、尾部循环等问题
- **并发处理**: 多处理器并行处理，充分利用GPU资源
- **增量保存**: 自动保存中间结果，防止数据丢失

### 🚀 性能优势
- **默认模型**: Qwen2.5-VL-3B-Instruct（仅3B参数，显存~8GB）
- **处理速度**: ~100-150 样本/分钟（4 GPUs, 8 processors）
- **向后兼容**: 完全兼容 2D 数据处理
- **灵活配置**: 支持命令行、环境变量、配置文件多种配置方式

### 📊 支持的数据格式

| 数据集 | 类型 | 图像 | 深度图 | 支持状态 |
|--------|------|------|--------|---------|
| RefSpatial 2D | choice_qa | 单图 | ❌ | ✅ |
| RefSpatial 2D | reasoning_template_qa | 单图 | ❌ | ✅ |
| RefSpatial 3D | choice_qa | 单图 | ✅ | ✅ |
| RefSpatial 3D | multi_view_qa | 多图 | ✅ | ✅ |
| RefSpatial 3D | visual_choice_qa | 单图+bbox | ✅ | ✅ |
| RefSpatial 3D | reasoning_template_qa | 单图 | ✅ | ✅ |
| RefSpatial 3D | vacant_qa | 单图 | ✅ | ✅ |

## 🚀 快速开始

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
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置并发参数
export NUM_PROCESSORS=8
export BATCH_SIZE=4
```

### 3. 运行数据清洗

#### 方式A: 2D数据清洗
```bash
# 编辑 quick_start.sh 中的路径配置
vim quick_start.sh

# 运行清洗
./quick_start.sh
```

#### 方式B: 3D数据清洗（单个数据集）
```bash
# 编辑 quick_start_3d.sh 中的路径配置
vim quick_start_3d.sh

# 运行清洗
./quick_start_3d.sh
```

#### 方式C: 批量处理所有3D数据（推荐）⭐
```bash
# 编辑 batch_clean_3d.sh 中的BASE_3D路径
vim batch_clean_3d.sh

# 批量运行
./batch_clean_3d.sh
```

### 4. 查看结果

清洗完成后，会生成以下文件：
- `*_cleaned.jsonl`: 清洗后的高质量数据（用于训练）
- `*_full.jsonl`: 完整评估结果（包含所有元数据）
- `*_system_prompt.txt`: 使用的评估提示词

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

### 基本用法

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

### 禁用启发式规则

```bash
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl \
    --no_heuristic  # 只使用VLM评分
```

## ⚙️ 配置参数

### 命令行参数

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
AgentCleaner/
├── main.py                     # 主程序入口
├── quick_start.sh              # 2D数据一键启动
├── quick_start_3d.sh           # 3D数据一键启动
├── batch_clean_3d.sh           # 批量处理所有3D数据 ⭐
├── requirements.txt            # 依赖列表
│
├── src/                        # 源代码
│   ├── config.py              # 配置管理
│   ├── base/                  # 核心组件
│   │   ├── vlm_agent.py       # VLM Agent（支持多图输入）
│   │   ├── processor.py       # 数据处理器（智能路径查找）
│   │   └── unified_manager.py # 统一管理器
│   └── utils/                 # 工具函数
│       ├── Data_Selector.py   # 数据选择器
│       └── Heuristic_Rules_Internvl2_5.py  # 启发式规则
│
├── scripts/                    # 辅助脚本
│   ├── extract_fields.py      # 字段提取工具
│   ├── batch_process.sh       # 多GPU批处理
│   └── analyze_results.py     # 结果分析
│
├── output/                     # 输出目录
│   ├── 2D/                    # 2D数据输出
│   ├── 2D_ex/                 # 2D提取字段输出
│   └── 3D/                    # 3D数据输出
│
├── examples/                   # 示例数据
│   ├── choice_qa_test.json    # 测试数据
│   ├── images/                # 测试图像
│   └── test_example.sh        # 测试脚本
│
└── logs/                       # 日志目录
```

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

## 📄 许可证

本项目遵循 MIT 许可证。

## 🙏 致谢

- Qwen2.5-VL 模型团队
- InternVL 模型团队
- 所有开源贡献者

---

**更新日期**: 2025-10-21  
**版本**: 2.0.0
