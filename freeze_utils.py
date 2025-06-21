def freeze_all_layers_except(model, target_index):
    """
    冻结除指定层之外的所有 Transformer 层。

    参数:
        model: Transformer 模型（如 GPT-Neo）
        target_index: 仅训练的目标层索引（0 ~ N-1）
    """
    for i, layer in enumerate(model.transformer.h):
        for param in layer.parameters():
            param.requires_grad = (i == target_index)

    # 同时冻结 embedding 和 lm_head 层（除非你想训练）
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
