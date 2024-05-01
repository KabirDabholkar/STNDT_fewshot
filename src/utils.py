import torch

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def print_gpu_memory_usage(_vars):
    print("GPU Memory Usage:")
    # print(_vars)
    for name,obj in _vars.items():
        if torch.is_tensor(obj) and obj.device.type == 'cuda':
            print('Object name:',name,", Memory Allocated:",sizeof_fmt(obj.element_size() * obj.nelement()))
            print()

def main():
    a = torch.zeros((10,10,10)).to('cuda:0')
    print_gpu_memory_usage(vars())
if __name__=='__main__':
    main()
    
    
