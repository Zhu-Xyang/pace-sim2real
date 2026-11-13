import unittest
from types import SimpleNamespace
from hip_hold_gains import apply_hip_hold_gains

class HipHoldTests(unittest.TestCase):
    def test_override_and_preserve(self):
        names = [f'J{i:02d}_{kind}_{side}' for i,(kind,side) in enumerate(
            [(k,s) for s in ('L','R') for k in ('HIP_PITCH','HIP_ROLL','HIP_YAW','KNEE_PITCH')])]
        a=SimpleNamespace(stiffness={'.*':10.}, damping={'.*':2.})
        apply_hip_hold_gains(a,names,'knee_pitch',[240,200,113],None)
        self.assertEqual(a.damping,{'.*':2.})
        for n in names:
            self.assertEqual(a.stiffness[n],dict(HIP_PITCH=240,HIP_ROLL=200,HIP_YAW=113,KNEE_PITCH=10)[n[4:-2]])
        old=a.stiffness.copy()
        for mode,kp in [('hip_pitch',[1,2,3]),('knee_pitch',[1,float('nan'),3]),('knee_pitch',[-1,2,3])]:
            with self.assertRaises(ValueError): apply_hip_hold_gains(a,names,mode,kp)
            self.assertEqual(a.stiffness,old)
    def test_defaults(self):
        a=SimpleNamespace(stiffness=10,damping=2)
        apply_hip_hold_gains(a,[],'hips')
        self.assertEqual(a.stiffness,10)
