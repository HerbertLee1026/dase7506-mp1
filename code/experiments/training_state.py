"""Portable, atomic training-state snapshots for future remote experiments."""

from pathlib import Path
import torch


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng(generator, device):
    result = {'cpu': torch.get_rng_state(), 'sampling': generator.get_state(),
              'device': str(device)}
    if str(device) == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state_all()
    elif str(device) == 'mps':
        result['mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(saved, generator, device):
    torch.set_rng_state(saved['cpu'])
    generator.set_state(saved['sampling'])
    if str(device) == saved['device']:
        if str(device) == 'cuda':
            if len(saved['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA device count changed; RNG replay cannot be exact.')
            torch.cuda.set_rng_state_all(saved['cuda'])
        elif str(device) == 'mps':
            torch.mps.set_rng_state(saved['mps'])
        return True
    return False


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    torch.save(cpu_tree(payload), temporary)
    temporary.replace(path)
