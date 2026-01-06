from config_loader import load_settings
import torch

def runtime_loader():
    settings = load_settings()
    run_on_device = settings['run_on_device']
    selected_device = settings['device_name']

    if not run_on_device:
        return torch.device('cpu')

    if selected_device.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(selected_device)
        else:
            print('No CUDA device detected. Falling back to CPU.')
            return torch.device('cpu')

    elif selected_device.startswith('mps'):
        if torch.backends.mps.is_available():
            return torch.device(selected_device)
        else:
            print('MPS not available. Falling back to CPU.')
            return torch.device('cpu')

    return torch.device('cpu')



if __name__ == "__main__":
    runtime_loader = runtime_loader()
