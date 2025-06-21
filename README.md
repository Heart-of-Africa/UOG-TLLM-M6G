# GPT-Neo FP32 分层训练项目（适配 6GB 显存）

该项目允许在显存非常有限的情况下（6GB）逐层训练模型。

## 使用方式

1. 安装依赖：
    pip install -r requirements.txt

2. 训练第 i 层（例如第 5 层）：
    python train_layer.py --layer 5

3. 所有层训练完后，可在 merge_checkpoints.py 中合并权重。

## 技术策略

- 每次只训练一个 Transformer 层
- 其他层全部冻结，节省显存
- 使用 FP32 精度
- 支持恢复与逐阶段检查

建议训练所有 24 层，每层输出会保存在 `./checkpoints/layer_XX/` 中。