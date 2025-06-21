def freeze_all_layers_except(model, target_index):
    for i, layer in enumerate(model.transformer.h):
        for param in layer.parameters():
            param.requires_grad = (i == target_index)
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False