# AgentCleaner - 阿里云EAS API使用说明

本文档介绍如何使用阿里云EAS部署的大模型API进行VQA数据清洗。

## ✨ 特性

- **无需本地GPU**: 通过API调用云端大模型，无需本地GPU资源
- **超大规模模型**: 支持使用 Qwen3-VL-235B-A22B-Instruct 等超大规模模型
- **简单易用**: 只需配置API地址和token即可使用
- **完全兼容**: 与本地模型模式完全兼容，代码无需修改

## 🚀 快速开始

### 1. 配置EAS服务信息

编辑 `quick_start_eas.sh` 文件，配置你的EAS服务信息：

```bash
# EAS服务基础URL
export EAS_BASE_URL="http://your-eas-service-url/api/predict/your-service-name"

# EAS认证token
export EAS_TOKEN="your-eas-token"

# 模型名称
export EAS_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
```

### 2. 配置数据路径

修改脚本中的数据路径：

```bash
# 2D数据配置
BASE_2D="/path/to/your/2D/data"
JSON_FILE="$BASE_2D/choice_qa_2D_qwenvl.json"
OUTPUT_FILE="./output/2D_eas/choice_qa_cleaned.jsonl"
ROOT_PATH="$BASE_2D/images/"
```

### 3. 运行清洗

```bash
./quick_start_eas.sh
```

## 📋 EAS API配置说明

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `AGENT_TYPE` | Agent类型，使用EAS时设为 `eas` | `local` |
| `EAS_BASE_URL` | EAS服务基础URL | - |
| `EAS_TOKEN` | EAS认证token | - |
| `EAS_MODEL_NAME` | 模型名称 | `Qwen3-VL-235B-A22B-Instruct` |
| `EAS_MAX_TOKENS` | 最大生成token数 | `256` |
| `EAS_TIMEOUT` | API请求超时时间（秒） | `60` |
| `NUM_PROCESSORS` | 并发处理器数量 | `4` |
| `BATCH_SIZE` | 批处理大小 | `2` |

### 并发配置建议

由于EAS API有调用频率限制，建议降低并发配置：

```bash
# 推荐配置
export NUM_PROCESSORS=4  # 并发处理器数量
export BATCH_SIZE=2      # 批处理大小
```

如果遇到API限流，可以进一步降低并发数：

```bash
export NUM_PROCESSORS=2
export BATCH_SIZE=1
```

## 💡 使用示例

### 示例1: 处理2D数据

```bash
#!/bin/bash

export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url/api/predict/service"
export EAS_TOKEN="your-token"
export EAS_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"

python main.py \
    --json_file /path/to/2D/data.json \
    --root_path /path/to/2D/images/ \
    --output ./output/2D_eas/cleaned.jsonl \
    --num_processors 4 \
    --batch_size 2
```

### 示例2: 处理3D数据（含深度图）

```bash
#!/bin/bash

export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url/api/predict/service"
export EAS_TOKEN="your-token"

python main.py \
    --json_file /path/to/3D/data.json \
    --root_path /path/to/3D/ \
    --output ./output/3D_eas/cleaned.jsonl
```

EAS Agent会自动处理RGB和深度图的多模态输入。

### 示例3: 命令行直接配置

也可以通过命令行设置环境变量：

```bash
AGENT_TYPE=eas \
EAS_BASE_URL="http://your-url" \
EAS_TOKEN="your-token" \
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output output.jsonl
```

## 🔧 API调用说明

### 请求格式

EAS Agent会自动将图像转换为base64编码，并构建OpenAI兼容的请求格式：

```json
{
  "model": "Qwen3-VL-235B-A22B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": "You are a multi-modal VQA Data Quality Assessment Expert..."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        },
        {
          "type": "text",
          "text": "评估问题..."
        }
      ]
    }
  ],
  "max_tokens": 256,
  "stream": false
}
```

### 多图输入支持

对于3D数据（RGB + Depth），会同时发送两张图像：

```json
{
  "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  // RGB
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  // Depth
    {"type": "text", "text": "评估问题..."}
  ]
}
```

## ⚠️ 注意事项

### 1. API限流

- EAS服务通常有QPS（每秒请求数）限制
- 建议设置较低的并发数（`NUM_PROCESSORS=2-4`）
- 如遇到429错误，降低并发配置

### 2. 超时设置

- 大模型推理可能需要较长时间
- 默认超时60秒，可根据实际情况调整
- 如经常超时，可增加 `EAS_TIMEOUT` 值

```bash
export EAS_TIMEOUT=120  # 增加到120秒
```

### 3. 成本控制

- API调用会产生费用
- 建议先用少量数据测试
- 可使用 `--start` 和 `--end` 参数分批处理

```bash
# 先处理前100条测试
python main.py \
    --json_file data.json \
    --root_path /path/to/images/ \
    --output test.jsonl \
    --end 100
```

### 4. 图像大小

- 图像会转换为base64编码上传
- 建议图像分辨率不要过高（如 1024x1024 以内）
- 过大的图像会增加传输时间和API成本

## 🆚 EAS vs 本地模型对比

| 特性 | EAS API | 本地模型 |
|------|---------|---------|
| GPU需求 | ❌ 不需要 | ✅ 需要 |
| 模型大小 | ✅ 可用超大模型 (235B) | ⚠️ 受限于显存 |
| 处理速度 | ⚠️ 受网络影响 | ✅ 本地快速 |
| 成本 | 💰 按调用计费 | 💰 硬件成本 |
| 并发能力 | ⚠️ 受API限流 | ✅ 本地控制 |
| 易用性 | ✅ 配置简单 | ⚠️ 需要环境配置 |

## 🐛 故障排查

### 问题1: 连接超时

```
Error: API request timeout
```

**解决方案**：
- 检查网络连接
- 增加超时时间: `export EAS_TIMEOUT=120`
- 确认EAS服务URL正确

### 问题2: 认证失败

```
Error: API请求失败 (状态码: 401)
```

**解决方案**：
- 检查 `EAS_TOKEN` 是否正确
- 确认token是否过期
- 联系管理员重新获取token

### 问题3: API限流

```
Error: API请求失败 (状态码: 429)
```

**解决方案**：
- 降低并发数: `export NUM_PROCESSORS=2`
- 降低batch size: `export BATCH_SIZE=1`
- 增加重试间隔

### 问题4: 图像编码失败

```
Error: 图像转base64失败
```

**解决方案**：
- 检查图像文件是否损坏
- 确认图像格式是否支持
- 尝试转换图像为标准JPEG/PNG格式

## 📞 技术支持

如有问题，请提供以下信息：
1. EAS服务URL（脱敏）
2. 错误日志
3. Python和依赖版本
4. 数据样本（如可能）

## 🔗 相关链接

- [阿里云PAI-EAS文档](https://help.aliyun.com/document_detail/113696.html)
- [Qwen-VL模型文档](https://github.com/QwenLM/Qwen-VL)
- [AgentCleaner主文档](README.md)

---

**更新日期**: 2025-10-24  
**版本**: 1.0.0

