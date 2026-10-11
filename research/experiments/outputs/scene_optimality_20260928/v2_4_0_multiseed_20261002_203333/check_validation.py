import sys
sys.dont_write_bytecode=True
from common import *
assert_imports()
from hrl_v2.checkpoint import identity
v=read(WORK/'validation/validation.json');binding=identity(WORK,CONFIG)
check(v['status']=='passed' and v['identity']==binding,'validation_status_identity')
write(OUTM/'validation_check.json',dict(status='passed',identity=binding,imports=assert_imports()))
