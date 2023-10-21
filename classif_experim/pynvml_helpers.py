'''
From: https://huggingface.co/docs/transformers/perf_train_gpu_one
'''

from pynvml import *


def print_gpu_utilization(tag=''):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied{(' @'+tag+' ') if tag else ''}: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()