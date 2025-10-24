# 更新日志

## [2.1.0] - 2025-10-24

### 🎉 新增功能

#### 阿里云EAS API支持
- **新增 `VLMAgentEAS` 类**: 通过阿里云EAS API调用云端大模型
  - 支持多模态输入（RGB + Depth）
  - 自动图像base64编码
  - OpenAI兼容的API格式
  - 完善的错误处理和重试机制

- **新增 `quick_start_eas.sh`**: 一键启动EAS模式数据清洗
  - 预配置Qwen3-VL-235B-A22B-Instruct模型
  - 优化并发配置以适应API限流
  - 支持2D和3D数据处理

- **新增 `test_eas_connection.py`**: EAS连接测试脚本
  - 验证API配置
  - 测试文本和图像推理
  - 详细的错误诊断

- **新增 `README_EAS.md`**: EAS使用完整文档
  - 快速开始指南
  - API配置说明
  - 并发优化建议
  - 故障排查指南
  - 成本控制建议

### 🔧 改进

- **配置系统增强**
  - 添加 `AGENT_TYPE` 配置项（`local` / `eas`）
  - 新增EAS相关配置项（URL、Token、模型名等）
  - 配置打印功能区分本地/EAS模式

- **统一管理器升级**
  - `UnifiedManager.initialize_agent()` 支持双模式
  - 根据 `AGENT_TYPE` 自动选择Agent类型
  - 完全向后兼容本地模型模式

- **文档更新**
  - README.md添加EAS使用说明
  - 更新项目结构图
  - 添加EAS vs 本地模型对比

### 📁 新增文件

```
src/base/vlm_agent_eas.py       # EAS Agent实现
quick_start_eas.sh              # EAS一键启动脚本
test_eas_connection.py          # EAS连接测试
README_EAS.md                   # EAS使用文档
CHANGELOG.md                    # 更新日志（本文件）
```

### 🔄 修改文件

```
src/config.py                   # 添加EAS配置
src/base/__init__.py            # 导出VLMAgentEAS
src/base/unified_manager.py    # 支持双模式Agent
README.md                       # 添加EAS说明
```

### 💡 使用示例

#### 本地模型模式（默认）
```bash
./quick_start.sh
```

#### EAS API模式（新增）
```bash
export AGENT_TYPE="eas"
export EAS_BASE_URL="http://your-eas-url"
export EAS_TOKEN="your-token"
./quick_start_eas.sh
```

### 🐛 Bug修复

无

### ⚠️ 破坏性变更

无 - 完全向后兼容

### 📊 性能影响

- EAS模式：无需本地GPU，速度取决于网络和API响应
- 本地模式：无变化

### 🔐 安全性

- EAS Token通过环境变量配置
- 不在代码中硬编码敏感信息
- 建议使用.env文件或环境变量管理

---

## [2.0.0] - 2025-10-21

### 主要功能
- 3D数据支持（RGB + Depth）
- 智能路径查找
- 并发处理优化
- 启发式规则增强

---

## [1.0.0] - 初始版本

### 核心功能
- VQA数据质量评估
- 多模态大模型集成
- 2D数据处理
- 基础启发式规则

