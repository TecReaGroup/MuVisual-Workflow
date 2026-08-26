# switcher

```txt
├── config/
│   ├── workflow.yaml       # 主配置
│   ├── step_name/ # 步骤名（不涉及顺序，具体顺序在 workflow.yaml 中定义）
│   │   ├── option_name.yaml # 模型配置
│   │   └── option_name.yaml
│   └── step_name/ # 步骤名
│       ├── option_name.yaml
│       └── option_name.yaml
```

- 支持懒加载，对于执行到的步骤，才会下载对应的模型文件
