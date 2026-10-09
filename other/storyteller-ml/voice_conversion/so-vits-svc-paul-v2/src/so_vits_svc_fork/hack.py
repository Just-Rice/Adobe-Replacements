import os
import torch
from logging import getLogger

LOG = getLogger(__name__)

def clear_cuda_cache():
    LOG.debug('torch.cuda.empty_cache() ...')
    torch.cuda.empty_cache()
    LOG.debug('torch.cuda.empty_cache() ... DONE')

def turn_off_jit_profiling():
    LOG.debug('torch._C._jit_set_profiling_mode(False) ...')
    torch._C._jit_set_profiling_mode(False)
    LOG.debug('torch._C._jit_set_profiling_mode(False) ... DONE')

def maybe_hacky_fix_before():
    LOG.debug("maybe_hacky_fix_before()")
    if os.environ.get('FIX_CACHE_BEFORE'):
        clear_cuda_cache()
    if os.environ.get('FIX_JIT_BEFORE'):
        turn_off_jit_profiling()

def maybe_hacky_fix_after():
    LOG.debug("maybe_hacky_fix_after()")
    if os.environ.get('FIX_CACHE_AFTER'):
        clear_cuda_cache()
    if os.environ.get('FIX_JIT_AFTER'):
        turn_off_jit_profiling()
