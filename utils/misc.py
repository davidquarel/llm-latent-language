
import warnings
import torch

def printd(*args, **kwargs):
    # Check if '__DEBUG__' is in the global namespace and if it is set to True
    if globals().get('__DEBUG__', False):
        print("DEBUG:", end=" ")
        print(*args, **kwargs)

def generate_derangement(n: int) -> torch.Tensor:
    indices = torch.arange(n)
    
    while True:
        perm = torch.randperm(n)
        if torch.all(perm != indices):  # Ensure no element stays in its original position
            return perm

def dearrange(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Returns a deranged version of the input tensor along the specified dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor of any shape.
        dim (int): Dimension to derange over.
    
    Returns:
        torch.Tensor: A deranged version of the input tensor along the specified dimension.
    """
    size = tensor.shape[dim]
    perm = generate_derangement(size)  # Get a derangement of the specified dimension
    perm = perm.to(tensor.device)
    return tensor.index_select(dim, perm)  # Apply derangement along the specified dimension


def autoreload():
    def is_notebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':  # Jupyter notebook/lab
                return True
            elif shell == 'TerminalInteractiveShell':  # IPython terminal
                return True
            return False
        except:
            return False

    if is_notebook():
        try:
            get_ipython().run_line_magic('load_ext', 'autoreload')
            get_ipython().run_line_magic('autoreload', '2')
        except:
            pass


def str_dict(d):
    # Create a formatted string from dictionary entries
    items = [f"{k}: {f'{v:.4f}' if isinstance(v, float) else v}" for k, v in d.items()]
    # Join all items in a single line
    return ', '.join(items)


def is_chinese_char(ch):
    """Check if a character is a Chinese character using a list of Unicode ranges and return range information.
    Now robust to invalid inputs."""
    try:
        c = ord(ch)
    except:
        warnings.warn("is_chinese_char recieved non-char input", category = RuntimeWarning)
        return False
    # List of tuples, each representing a range of Chinese character code points with labels
    unicode_ranges = [
        (0x4E00, 0x9FFF, 'Common'),
        (0x3400, 0x4DBF, 'Extension A'),
        (0x20000, 0x2A6DF, 'Extension B'),
        (0x2A700, 0x2B73F, 'Extension C'),
        (0x2B740, 0x2B81F, 'Extension D'),
        (0x2B820, 0x2CEAF, 'Extension E'),
        (0x2CEB0, 0x2EBEF, 'Extension F'),
        (0x30000, 0x3134F, 'Extension G'),
        (0x31350, 0x323AF, 'Extension H'),
        (0xF900, 0xFAFF, 'CJK Compatibility Ideographs'),
        (0x2F800, 0x2FA1F, 'CJK Compatibility Ideographs Supplement')
    ]
    
    # Check if the character's code point falls within any of the ranges and return the range label
    for start, end, label in unicode_ranges:
        if start <= c <= end:
            return True
    return False

def ci(data, dim=0, debug = False):
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    sem95 = 1.96 * std / (len(data)**0.5) 
    if debug:
        print(f"{mean} ± {sem95}")
    return mean, sem95

import numpy as np
from scipy import stats

def wilson_ci(correct, total, confidence=0.95):
    """
    Calculate Wilson score interval, which is more reliable than normal approximation,
    especially for extreme proportions or small sample sizes.
    
    Parameters:
    correct (int): Number of correct predictions
    total (int): Total number of predictions
    confidence (float): Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    tuple: (accuracy, lower_bound, upper_bound)
    """
    accuracy = correct / total
    
    # Critical value for the desired confidence level
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate components of Wilson score interval
    denominator = 1 + z**2/total
    center_adjusted_probability = (accuracy + z**2/(2*total))/denominator
    adjusted_standard_error = z * np.sqrt((accuracy*(1 - accuracy) + z**2/(4*total))/total)/denominator
    
    lower_bound = max(0, center_adjusted_probability - adjusted_standard_error)
    upper_bound = min(1, center_adjusted_probability + adjusted_standard_error)
    
    return accuracy, lower_bound, upper_bound