from simple_patch.simple_patch import *
from fire import Fire

import inspect, io, termcolor as _tm, pprint as _pp
def formatted_file_and_lineno(depth=1): frame_stack = inspect.getouterframes(inspect.currentframe()); parrent = frame_stack[depth]; return f'{parrent.filename}:{parrent.lineno} $> '
def cpprint(obj, color='white'): buffer = io.StringIO(); _pp.pprint(obj, buffer); _tm.cprint(buffer.getvalue(), color);
def prepend_lineno(*obj, depth=1): return f'{formatted_file_and_lineno(depth + 1)}{" ".join(map(str, obj))}'
def lineno_print(*obj):
    if obj == '' or len(obj) == 0:
        _print(*obj)
    else:
        _print(prepend_lineno(*obj, depth=2))
def lineno_pprint(obj): _print(formatted_file_and_lineno(depth=2), end=''); _pp.pprint(obj)
def lineno_cprint(obj, color='white'): _tm.cprint(prepend_lineno(obj, depth=2), color)
def lineno_cpprint(obj, color='white'): _print(formatted_file_and_lineno(depth=2), end=''); _cpprint(obj, color);
_print = print; print = lineno_print; cprint = lineno_cprint; pprint = lineno_pprint; _cpprint = cpprint; cpprint = lineno_cpprint;

if __name__ == '__main__':
    test_extension()
    test_new_hashing_function()
    test_vector_space_function()
    test_aggregation()
    #algo_v1(pickle_me=True)
