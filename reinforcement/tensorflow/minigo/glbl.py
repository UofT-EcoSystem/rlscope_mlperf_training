# Define globals here.

from profiler.profilers import Profiler
from profiler import profilers

prof = None
def get_profiler():
    global prof
    # if prof is None:
    #     prof = profilers.Profiler(*args, **kwargs)
    return prof

def init_profiler(*args, **kwargs):
    global prof
    assert prof is None
    prof = profilers.Profiler(*args, **kwargs)
    return prof
