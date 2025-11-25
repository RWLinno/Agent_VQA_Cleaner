# 点位验证系统说明文档

## 📖 功能概述

本系统专门用于洗衣机数据集的点位验证，包含以下核心功能：

### Step 1: VLM点位生成与对比
- ✅ 使用235B模型根据图片和问题生成新的点坐标（P2）
- ✅ 将P2与原始标注点（P1）进行对比
- ✅ 计算两个点之间的欧氏距离
- ✅ 判断点位是否匹配（基于距离阈值）
- 🆕 **改进的Prompt**：强调精确定位，明确要求标注中心点

### Step 2: PointQA验证
- ✅ 使用PointQA问答方式验证原始点P1的准确性
- ✅ 让模型判断"点[x, y]是否正确对应label"
- ✅ 返回验证结果（正确/不正确）和置信度分数

### Step 3: Label提取
- ✅ 从问题文本中自动提取label信息
- 🆕 **增强的正则表达式**：支持更多问题格式
  - 引号包围：`Point to "power button"`
  - Object格式：`Object: control knob`
  - 各种动词：`Point out/Generate a list/Where are...`
  - If句式：`If there is a power button present...`
  - 常见模式回退：从预定义列表匹配

## 🎯 最新改进 (v1.1)

### 改进1: 更精确的点位定位Prompt
**问题**：原始P2点位准确率偏低（63%匹配，36% PointQA正确）

**解决方案**：改进VLM生成点位的prompt，增加以下指引：
- 明确要求定位到中心点（CENTER POINT）
- 针对不同组件类型给出具体指引（按钮、旋钮、面板、把手等）
- 要求提供精确的像素坐标
- 提供输出格式示例

### 改进2: 增强的Label提取
**问题**：许多label提取失败，显示为"unknown object"

**解决方案**：扩展正则表达式模式：
- 支持`Object: xxx`格式
- 支持更多动词模式（Point out, Generate a list, Where are等）
- 添加常见label的回退匹配机制
- 改进空格和标点处理

## 🚀 快速开始

### 1. 环境配置

```bash
# 设置EAS API（使用235B模型）
export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url"
export EAS_TOKEN="your-token"
export EAS_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct-BF16"
```

### 2. 快速测试（推荐先运行）

```bash
# 测试10条数据，快速验证功能
cd /mnt/users/rwl/Agent_VQA_Cleaner

python main_point_verification.py \
    --input /mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all/train_single_label_float.jsonl \
    --root_path /mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all \
    --output output/test_verification.jsonl \
    --num_workers 2 \
    --end 10

# 生成测试结果可视化
python visualize_point_verification.py \
    --input output/test_verification.jsonl \
    --output output/test_visualization.html \
    --filter all
```

### 3. 处理完整数据集

#### 方式A: 使用一键脚本（推荐）⭐

```bash
# 编辑脚本配置（修改END_INDEX控制处理数量）
vim process_washing_machine_point_verification.sh

# 运行处理
./process_washing_machine_point_verification.sh
```

**脚本配置说明**：
- `END_INDEX=100`：处理前100条（测试用）
- `END_INDEX=0`：处理全部数据（生产用）
- `NUM_WORKERS=8`：并发数，根据API限制调整

#### 方式B: 手动运行

```bash
# Step 1: 点位验证
python main_point_verification.py \
    --input /path/to/train_single_label_float.jsonl \
    --root_path /path/to/data_dir \
    --output output/verification_results.jsonl \
    --num_workers 8 \
    --distance_threshold 50

# Step 2: 生成可视化
python visualize_point_verification.py \
    --input output/verification_results.jsonl \
    --output output/visualization.html \
    --sample_size 50 \
    --filter all
```

## 📂 数据格式

### 输入格式（JSONL）

```json
{
  "id": "LG_1_LG_FCY10R4W_LG_(1).png_washing_machine_power_button_1",
  "messages": [
    {
      "role": "user",
      "content": "<image>\nPoint to all occurrences of \"power button\". The power button is the button that turns the washing machine on or off.\nOutput in JSON formate"
    },
    {
      "role": "assistant",
      "content": "```json\n[\n\t{\"point_2d\": [420.91, 263.16], \"label\": \"power button\"}\n]\n```"
    }
  ],
  "images": ["/path/to/image.png"]
}
```

### 输出格式（JSONL）

```json
{
  "id": "sample_id",
  "question": "Point to all occurrences of \"power button\"...",
  "answer": "{\"point_2d\": [420.91, 263.16], ...}",
  "image_path": "image.png",
  "full_image_path": "/full/path/to/image.png",
  "label": "power button",
  
  "point_original": {
    "coordinates": [420.91, 263.16],
    "count": 1,
    "all_points": [[420.91, 263.16]]
  },
  
  "point_inferred": {
    "coordinates": [425.0, 265.0],
    "vlm_response": "[425.0, 265.0]",
    "success": true
  },
  
  "point_comparison": {
    "distance": 5.12,
    "match": true,
    "threshold": 50.0
  },
  
  "pointqa_verification": {
    "is_correct": true,
    "confidence": 8.5,
    "response": "CORRECT | Confidence: 8.5/10 | The point accurately identifies..."
  },
  
  "status": "success",
  "processor_id": 0
}
```

## ⚙️ 配置参数

### 主程序参数（main_point_verification.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入JSONL文件路径 | 必需 |
| `--root_path` | 图像根目录 | 必需 |
| `--output` | 输出JSONL文件路径 | 必需 |
| `--num_workers` | 并行worker数量 | 4 |
| `--distance_threshold` | 点位距离阈值（像素） | 50.0 |
| `--start` | 起始索引 | 0 |
| `--end` | 结束索引 | None |
| `--log_level` | 日志级别 | INFO |
| `--log_file` | 日志文件路径 | None |

### 可视化参数（visualize_point_verification.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入JSONL文件路径 | 必需 |
| `--output` | 输出HTML文件路径 | 必需 |
| `--sample_size` | 抽样数量 | 50 |
| `--filter` | 过滤模式 | all |
| `--seed` | 随机种子 | 42 |

**过滤模式选项**：
- `all`: 显示所有样本
- `matched`: 只显示点位匹配的样本
- `mismatched`: 只显示点位不匹配的样本
- `correct`: 只显示PointQA验证正确的样本
- `incorrect`: 只显示PointQA验证不正确的样本

## 📊 可视化说明

可视化HTML会显示：

1. **总览统计**
   - 总样本数
   - 点位匹配数/匹配率
   - PointQA正确数/正确率
   - 平均距离和平均置信度

2. **每个样本的详细信息**
   - 图片上标注P1（红色）和P2（绿色）
   - P1和P2之间的连线（黄色虚线）
   - Label信息
   - 问题文本
   - 点位对比信息（坐标、距离、是否匹配）
   - PointQA验证结果（正确性、置信度、模型响应）

### 图例说明
- 🔴 **P1（红色）**: 原始标注点
- 🟢 **P2（绿色）**: VLM推理生成的点
- 💛 **黄色虚线**: 连接P1和P2，越短说明越准确

## 🔍 核心算法

### 1. Label提取算法（v1.1增强版）

```python
# 模式1: 引号包围的label
pattern1 = r'"([^"]+)"'

# 模式2: Object: xxx 格式 (新增)
pattern_object = r'Object:\s*([^\n]+)'

# 模式3: 各种动词模式 (增强)
pattern_verbs = r'(?:Locate|Find|Point(?:\s+to)?(?:\s+out)?|Show\s+me(?:\s+where)?|Generate\s+a\s+list.*?where|Where\s+are|Can\s+you\s+point\s+out)\s+(?:each|every|all|a|an|the)?\s*([a-z\s/\-]+?)...'

# 模式4: If there (is|are) (a|any) xxx (增强)
pattern_if = r'If\s+there\s+(?:is|are)\s+(?:a|any)\s+([a-z\s/\-]+?)...'

# 模式5: 常见label回退匹配 (新增)
common_labels = ['power button', 'start button', 'control panel', 'control knob', ...]
```

### 2. 点位生成Prompt（v1.1改进版）

**改进重点**：强调精确定位中心点，提供明确指引

```
Task: Identify the exact location of the "{label}" in this washing machine image.

Instructions:
1. Carefully locate the CENTER POINT of the {label}
2. If it's a button, point to its center
3. If it's a control panel/display area, point to the center of the display
4. If it's a knob, point to its center
5. If it's a handle, point to the middle of the handle
6. Provide PRECISE pixel coordinates in the format [x, y]

Output format: [x, y]
Example: [425, 263]

Coordinate:
```

**改进效果**：
- 更明确的任务描述
- 针对不同组件类型的具体指引
- 强调精确性和格式要求
- 提供输出示例

### 3. 点位对比算法

```python
# 欧氏距离计算
distance = sqrt((p1[0] - p2[0])^2 + (p1[1] - p2[1])^2)

# 判断匹配
match = (distance <= threshold)  # 默认threshold=50px
```

### 4. PointQA验证Prompt

```
I want to verify if the point [x, y] correctly identifies the "{label}" in this image.

Please answer with one of the following:
- "CORRECT" if the point accurately identifies the label
- "INCORRECT" if the point does not correctly identify the label
- "UNCERTAIN" if you cannot determine

Then provide a confidence score from 0 to 10 and a brief explanation.

Format: [CORRECT/INCORRECT/UNCERTAIN] | Confidence: X/10 | Explanation: ...
```

## 📈 性能基准

### 处理速度（8 workers，235B模型）
- 单个样本处理时间：~5-8秒（包含2次VLM推理）
- 并行处理速度：~60-90 样本/分钟

### API调用次数
每个样本需要调用2次VLM API：
1. 生成新的点坐标（P2）
2. PointQA验证原始点（P1）

## 🎯 使用建议

### 1. 测试少量数据
先用少量数据测试，确保配置正确：

```bash
# 只处理前10条数据
python main_point_verification.py \
    --input data.jsonl \
    --root_path /path/to/images \
    --output test_output.jsonl \
    --num_workers 2 \
    --start 0 \
    --end 10
```

### 2. 调整距离阈值

根据数据集特点调整阈值：
- **严格匹配**: 阈值 = 20-30px
- **一般匹配**: 阈值 = 50px（默认）
- **宽松匹配**: 阈值 = 80-100px

### 3. 分批处理大数据集

```bash
# 处理第0-500条
python main_point_verification.py --start 0 --end 500 ...

# 处理第500-1000条
python main_point_verification.py --start 500 --end 1000 ...
```

### 4. 可视化过滤

重点查看问题样本：

```bash
# 只查看点位不匹配的样本
python visualize_point_verification.py \
    --input results.jsonl \
    --output vis_mismatch.html \
    --filter mismatched

# 只查看PointQA验证失败的样本
python visualize_point_verification.py \
    --input results.jsonl \
    --output vis_incorrect.html \
    --filter incorrect
```

## 🐛 故障排查

### 问题1: EAS API连接失败

```
Error: API请求失败
```

**解决方案**：
- 检查 `EAS_BASE_URL` 和 `EAS_TOKEN` 是否正确
- 确认网络可以访问EAS服务
- 运行测试脚本验证连接：`python test_point_verification.py`

### 问题2: 图像文件未找到

```
Image not found: /path/to/image.png
```

**解决方案**：
- 检查 `--root_path` 是否正确
- 确认图像文件在 `train/` 子目录下
- 检查JSONL中的图像路径是否正确

### 问题3: Label提取失败

```
无法从问题中提取label
```

**解决方案**：
- 检查问题文本格式
- 可能需要添加新的正则表达式模式
- 查看 `point_verification_processor.py` 中的 `extract_label_from_question` 方法

### 问题4: VLM未返回有效点坐标

```
模型未返回有效点坐标
```

**解决方案**：
- 检查VLM响应内容
- 可能需要调整prompt
- 增加 `max_tokens` 参数

## 📝 输出文件说明

运行脚本后会生成以下文件：

```
output/washing_machine_point_verification/
├── train_single_label_float_verification_results.jsonl  # 完整验证结果
├── train_single_label_float_verification.log           # 处理日志
└── train_single_label_float_visualization.html         # 可视化报告
```

## 🔗 相关文件

- `src/base/point_verification_processor.py` - 核心处理器
- `main_point_verification.py` - 主程序入口
- `visualize_point_verification.py` - 可视化工具
- `test_point_verification.py` - 功能测试脚本
- `process_washing_machine_point_verification.sh` - 一键处理脚本

## 🆕 与原系统的区别

| 功能 | 原系统 | 新系统 |
|------|--------|--------|
| 主要任务 | VQA数据质量评估 | 点位验证与对比 |
| 评估方式 | 质量打分（0-10分） | 点位准确性验证 |
| 输出内容 | 过滤后的高质量数据 | 点位对比结果 |
| 可视化 | 显示图片和QA | 显示P1/P2对比 |
| 应用场景 | 数据清洗 | 标注质量检查 |

## 📞 常见问题

**Q: 处理速度慢怎么办？**

A: 增加 `--num_workers` 参数，但要注意EAS API的并发限制。

**Q: 如何只验证特定类型的label？**

A: 在处理前先过滤JSONL文件，只保留特定label的数据。

**Q: P1和P2距离很近但PointQA说不正确？**

A: 可能是因为：
1. 点位虽然接近但位置仍然不准确
2. 模型对该label的理解与实际标注有差异
3. 图像中该label存在多个实例

**Q: 可以用本地模型吗？**

A: 当前版本主要针对EAS API优化，如需本地模型支持请修改代码。

## 📝 版本历史

### v1.1 (2025-11-11)

**改进内容**：
1. ✅ **增强Label提取**：支持更多问题格式，减少"unknown object"错误
2. ✅ **改进点位生成Prompt**：强调中心点定位，提高P2点位准确性
3. ✅ **文档整理**：删除冗余文档，集中维护单一README

**预期效果**：
- Label提取成功率：显著提升
- 点位匹配率：从63%→75%+（预期）
- PointQA正确率：从36%→50%+（预期）

### v1.0 (2025-11-11)

**初始功能**：
- ✅ VLM点位生成与对比
- ✅ PointQA验证
- ✅ Label自动提取
- ✅ HTML可视化
- ✅ 并发处理

## 📄 许可证

遵循项目根目录的LICENSE文件。

---

**最后更新**: 2025-11-11  
**当前版本**: v1.1

