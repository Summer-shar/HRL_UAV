# D9 import-checked entry for unchanged validation and train_upper CLI modules.
import sys
sys.dont_write_bytecode=True
import runpy
from common import assert_imports
assert_imports()
module=sys.argv[1];sys.argv=[module]+sys.argv[2:]
runpy.run_module(module,run_name='__main__')
