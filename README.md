This project is licensed for personal, non-commercial use only.
Any redistribution or commercial usage is strictly prohibited.

# OVERTURE BP16 分层训练项目（适配 6GB 显存）

该项目允许在显存非常有限的情况下（最低6GB）逐层训练模型。

## 使用方式

1. 安装依赖：
    pip install -r requirements.txt

2. 训练第 i 层（例如第 5 层）：
    python train_layer.py --layer 5

3. 训练完 i 层，即可尝试部署 i 层
    python infer_layer0.py

5. 所有层训练完后，可在 merge_checkpoints.py 中合并权重。

## 技术策略

- 每次只训练一个 Transformer 层
- 其他层全部冻结，节省显存
- 使用 BF16 精度
- 支持恢复与逐阶段检查

建议训练所有 24 层，每层输出会保存在 `./checkpoints/layer_XX/` 中。

# LLM FP32 Layer-by-Layer Training Program (for 6GB video memory)

This project allows to train models layer by layer with very limited video memory (minimum 6GB).

## Usage

1. Install the dependencies: 
 pip install -r requirements.txt

2. Train layer i (e.g. layer 5): 
 python train_layer.py --layer 5

3. After all layers are trained, merge the weights in merge_checkpoints.py.

## Technical Strategy

- Train only one Transformer layer at a time
- Freeze all other layers to save memory
- Use BF16 precision
- Supports recovery and stage-by-stage checking

It is recommended to train all 24 layers, and the output of each layer is saved in `. /checkpoints/layer_XX/`.
