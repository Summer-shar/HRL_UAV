"""Exact numeric comparison after Python-float conversion; no tolerance."""
from numbers import Real
from j_common import HardCheckError
def exact_equal(left,right,label,record=False):
 facts=[]
 def fail(path,a,b,why):
  raise HardCheckError(str(dict(check=label,field=path,reason=why,candidate_repr=repr(a),reference_repr=repr(b))))
 def walk(a,b,path):
  if isinstance(a,dict) and isinstance(b,dict):
   if set(a)!=set(b):fail(path,sorted(a,key=str),sorted(b,key=str),'keys')
   for k in sorted(a,key=str):walk(a[k],b[k],path+'.'+str(k))
  elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
   if len(a)!=len(b):fail(path,len(a),len(b),'length')
   for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'.'+str(i))
  elif isinstance(a,Real) and isinstance(b,Real):
   x,y=float(a),float(b)
   if record:facts.append(dict(field=path,candidate_repr=repr(a),reference_repr=repr(b),candidate_float_repr=repr(x),reference_float_repr=repr(y),candidate_hex=x.hex(),reference_hex=y.hex()))
   if x!=y:fail(path,x,y,'numeric_exact_float')
  elif a!=b:fail(path,a,b,'value')
 walk(left,right,label)
 return dict(passed=True,numeric_reprs=facts,comparison='Python float ==; zero tolerance')
