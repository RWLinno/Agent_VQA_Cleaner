# Agent_VQA_Cleaner Examples

本目录包含Agent_VQA_Cleaner的示例数据和测试脚本。

## 文件说明

### 测试数据
- `choice_qa_test.json`: 测试用的JSON数据文件
- `images/`: 测试图像目录
  - `00056c76aa3dd52a.jpg`: 示例图像

### 测试脚本
- `test_example.sh`: 测试运行脚本，用于验证系统功能

### 配置示例
- `example_config.sh`: 配置示例脚本

### 输出目录
- `output/`: 测试输出结果目录
  - `test_result.jsonl`: 清洗后的测试结果
  - `test_result_full_full.jsonlll`: 完整测试结果
  - `test_result_system_prompt.txt`: 系统提示词

## 快速测试

运行以下命令进行快速测试：

```bash
cd examples
bash test_example.sh
```

该脚本会：
1. 使用示例数据进行测试
2. 生成清洗结果
3. 显示统计信息

## 数据格式

### 输入格式 (choice_qa_test.json)

```json
{
  "id": "sample_001",
  "image": ["00056c76aa3dd52a.jpg"],
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

### 输出格式 (test_result.jsonl)

```json
{
  "question": "问题文本",
  "answer": "答案文本",
  "image_path": "00056c76aa3dd52a.jpg",
  "score": 8
}
```

## 注意事项

- 示例数据仅用于测试系统功能
- 实际使用时请替换为自己的数据集
- 确保图像路径正确且文件存在
