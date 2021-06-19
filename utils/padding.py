
def get_padding(kernel_size, mode='same'):
    if mode == 'same':
        return (kernel_size // 2, kernel_size // 2)
    elif mode == 'valid':
        return (0, 0)
    else:
        raise NotImplementedError("Unknown padding mode")