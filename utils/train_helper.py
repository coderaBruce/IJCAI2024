
def reduce_learning_rate(optimizer, factor=1.0, min_lr=1e-6):
    for param_group in optimizer.param_groups:
        reduced_lr = max(param_group["lr"] * factor, min_lr)
        param_group["lr"] = reduced_lr
    return reduced_lr

def reset_learning_rate(config_dict, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = config_dict['learning_rate']
    return 