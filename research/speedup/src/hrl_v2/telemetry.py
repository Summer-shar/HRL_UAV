"""Read-only resource and compact episode statistics."""
import os
import torch


def resources():
    result=dict(cuda_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                cuda_reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        value=Memory(); value.cb=ctypes.sizeof(value)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.WinDLL('psapi',use_last_error=True)
        api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
        if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb): raise ctypes.WinError(ctypes.get_last_error())
        result.update(ram_bytes=value.WorkingSetSize,ram_peak_bytes=value.PeakWorkingSetSize)
    else:
        result.update(ram_bytes=None,ram_peak_bytes=None)
    return result


def compact(metrics):
    keys=('truncated_delay_s','weighted_completed_delay_s','stage_delay_s','unfinished_parents',
          'oldest_unfinished_age_s','cost_per_s','energy_j','arrived_parents','holding_task_s','seconds','physical_steps')
    result={k:metrics[k] for k in keys}
    result['priority']=[{k:v for k,v in row.items() if k not in ('completed_delays_s','cutoff_ages_s')} for row in metrics['priority']]
    return result
