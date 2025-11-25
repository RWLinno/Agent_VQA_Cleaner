# 点位验证功能更新说明

## 📅 更新日期：2025-11-05

## ✨ 更新内容

### 1. HTML可视化快速跳转功能修复

**问题：** 快速跳转链接无法定位到对应的样本卡片

**修复：** 
- 在 `visualize_samples.py` 中为每个样本div添加了id属性：`id="sample-{idx}"`
- 现在快速跳转链接 `<a href="#sample-{idx}">` 可以正确定位

**文件修改：**
```python
# visualize_samples.py Line 629
<div class="sample" id="sample-{idx}">  # 添加了id属性
```

### 2. 洗衣机数据集点位验证功能

**背景：** 
洗衣机数据集是点定位任务，答案中包含坐标点。为了确保数据质量，需要验证答案中的点位是否准确。

**解决方案：**
通过让模型对同一图片重新推理，比较原答案点位（P1）和模型推理点位（P2）的距离，如果差异过大则扣分。

#### 2.1 清洗脚本改进 (`clean_washing_machine_eas.sh`)

**新增配置：**
```bash
# 数据文件（使用单标签浮点数据）
TRAIN_FILE="train_single_label_float.jsonl"
TEST_FILE="test_single_label_float.jsonl"

# 点位验证配置
export ENABLE_POINT_VERIFICATION=true  # 启用点位验证
export POINT_DISTANCE_THRESHOLD=50     # 点位距离阈值（像素）
export POINT_VERIFICATION_PENALTY=3    # 点位不准扣分
```

**配置说明：**
- `ENABLE_POINT_VERIFICATION`: 是否启用点位验证（true/false）
- `POINT_DISTANCE_THRESHOLD`: 允许的最大偏差距离（像素），超过此值则认为不准确
- `POINT_VERIFICATION_PENALTY`: 点位不准时的扣分

#### 2.2 Processor核心功能 (`src/base/processor.py`)

**新增方法：**

1. **`_extract_point_coordinates(text)`**
   - 从文本中提取点坐标
   - 支持格式：`[x, y]`、`[x,y]`、`[ x , y ]`
   - 返回坐标列表：`[(x1, y1), (x2, y2), ...]`

2. **`_calculate_point_distance(p1, p2)`**
   - 计算两个点之间的欧式距离
   - 公式：`sqrt((x1-x2)^2 + (y1-y2)^2)`

3. **`_verify_point_accuracy(question, answer, image_path, depth_path)`**
   - 验证答案中点位的准确性
   - 工作流程：
     1. 提取原答案中的第一个点位 P1
     2. 构建验证prompt让模型重新推理
     3. 从模型响应中提取推理点位 P2
     4. 计算 P1 和 P2 的距离
     5. 如果距离 > 阈值，返回扣分和原因

**验证流程：**

```
原答案: "```json\n[{\"point_2d\": [420.91, 263.16], \"label\": \"开关\"}]\n```"
       ↓ 提取P1
P1 = (420.91, 263.16)
       ↓ 模型重新推理
模型响应: "The location is [425.3, 268.5]"
       ↓ 提取P2
P2 = (425.3, 268.5)
       ↓ 计算距离
distance = sqrt((420.91-425.3)^2 + (263.16-268.5)^2) = 6.8px
       ↓ 判断
distance (6.8px) < threshold (50px) → 通过验证
```

**集成位置：**
- `process_single()` 方法：在启发式规则后、VLM评分前调用
- `process_batch()` 方法：同样在启发式规则后调用

## 📊 使用示例

### 启用点位验证清洗数据

```bash
cd /mnt/users/rwl/Agent_VQA_Cleaner

# 方式1：使用脚本（已配置好）
./clean_washing_machine_eas.sh

# 方式2：手动配置环境变量
export ENABLE_POINT_VERIFICATION=true
export POINT_DISTANCE_THRESHOLD=50
export POINT_VERIFICATION_PENALTY=3

python main.py \
    --json_file /path/to/data.json \
    --root_path /path/to/images \
    --output output/cleaned.jsonl \
    --score_threshold 6
```

### 禁用点位验证

```bash
export ENABLE_POINT_VERIFICATION=false
./clean_washing_machine_eas.sh
```

### 调整验证参数

```bash
# 更严格的验证（距离阈值30px，扣5分）
export POINT_DISTANCE_THRESHOLD=30
export POINT_VERIFICATION_PENALTY=5

# 更宽松的验证（距离阈值100px，扣2分）
export POINT_DISTANCE_THRESHOLD=100
export POINT_VERIFICATION_PENALTY=2
```

## 📈 效果示例

### 过滤原因示例

**点位准确（不扣分）：**
```json
{
  "question": "Show me where the 开关 is.",
  "answer": "[{\"point_2d\": [420.91, 263.16]}]",
  "score": 8,
  "filtered": false,
  "filter_reason": null
}
```

**点位不准（扣分）：**
```json
{
  "question": "Show me where the 开关 is.",
  "answer": "[{\"point_2d\": [420.91, 263.16]}]",
  "score": 4,
  "original_score": 7,
  "heuristic_penalty": 3,
  "heuristic_flags": [
    "Point location mismatch: original (420.91, 263.16) vs inferred (520.5, 350.2), distance=115.3px (>50px)"
  ],
  "filtered": true,
  "filter_reason": "Quality issues: Point location mismatch: original (420.91, 263.16) vs inferred (520.5, 350.2), distance=115.3px (>50px) | Original VLM score 7/10 reduced to 4/10 after quality penalties"
}
```

## 🔍 技术细节

### 为什么只验证第一个点？

```python
# 只取第一个点进行验证（假设是主要定位点）
p1 = points_original[0]
```

**原因：**
1. **效率考虑**：验证每个点都需要额外的模型推理，会显著增加处理时间
2. **代表性**：第一个点通常是主要定位点，能代表答案质量
3. **避免误判**：多个点可能有不同的语义，逐一验证容易产生误判

如果需要验证所有点，可以修改代码循环处理：
```python
for i, p1 in enumerate(points_original):
    # 验证每个点...
```

### 验证失败时的处理

```python
except Exception as e:
    logger.error(f"点位验证过程出错: {e}")
    # 验证失败时不扣分，避免误判
    return 0, ""
```

**设计原则：**
- 验证过程出错时不扣分（宁可漏过，不可错杀）
- 记录错误日志便于调试
- 确保验证功能不会影响正常的数据清洗流程

### 验证Prompt设计

```python
verification_prompt = f"{question}\nPlease provide the exact location as a coordinate point in the format [x, y]."
```

**设计要点：**
- 保留原问题，确保语义一致
- 要求模型返回坐标格式，便于提取
- 不包含原答案，避免模型受影响

## ⚙️ 配置建议

### 不同场景的参数设置

| 场景 | THRESHOLD | PENALTY | 说明 |
|------|-----------|---------|------|
| 精确定位任务 | 30 | 5 | 严格要求，大偏差重罚 |
| 一般定位任务 | 50 | 3 | 默认配置，平衡精度和召回 |
| 粗略定位任务 | 100 | 2 | 宽松要求，只过滤明显错误 |
| 高分辨率图片 | 80 | 3 | 图片大，允许更大偏差 |
| 低分辨率图片 | 30 | 4 | 图片小，要求更高精度 |

### 性能影响

启用点位验证会增加处理时间：
- **额外推理次数**：每个含点位的样本 +1 次推理
- **预估时间增加**：约 50-100%（取决于点位数据比例）

建议：
- 对于不含点位的数据集，建议禁用此功能
- 对于含少量点位的数据集，可以启用
- 对于全是点位数据的数据集，考虑降低采样率或并发数

## 🐛 故障排查

### 问题1：点位验证总是失败

**可能原因：**
- 阈值设置过小
- 模型推理不稳定

**解决方案：**
```bash
# 增大阈值
export POINT_DISTANCE_THRESHOLD=100

# 查看日志
tail -f logs/cleaning.log | grep "点位验证"
```

### 问题2：验证过程很慢

**可能原因：**
- 每个样本都需要额外推理

**解决方案：**
```bash
# 方案1：禁用点位验证
export ENABLE_POINT_VERIFICATION=false

# 方案2：使用更快的模型（本地模型）
export AGENT_TYPE="local"

# 方案3：增加并发数
export NUM_PROCESSORS=32
```

### 问题3：验证结果不稳定

**可能原因：**
- 模型推理存在随机性
- 问题描述不够明确

**解决方案：**
- 修改验证prompt，使其更明确
- 多次验证取平均值（需修改代码）

## 📝 总结

本次更新主要包括：

1. ✅ **修复HTML可视化快速跳转** - 添加锚点id
2. ✅ **实现点位验证功能** - 通过重新推理验证答案准确性
3. ✅ **更新清洗脚本** - 使用正确的数据文件和验证配置
4. ✅ **详细的配置选项** - 灵活控制验证行为

这些改进让数据清洗系统能够：
- 更好地处理点定位任务数据
- 自动检测和过滤不准确的标注
- 提供更详细的过滤原因
- 保持高可配置性和灵活性

---

**版本：** 2.2.1  
**更新日期：** 2025-11-05  
**维护者：** Agent VQA Cleaner Team

