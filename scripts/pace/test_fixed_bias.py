"""CPU-only check of fixed-bias optimization and sampled parameters."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

class FixedBiasTests(unittest.TestCase):
    def test_fixed_bias_excluded_and_preserved(self):
        path=Path(__file__).resolve().parents[2]/'source/pace_sim2real/pace_sim2real/optim/cma_es.py'
        spec=importlib.util.spec_from_file_location('fixed_bias_optimizer',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            bounds=torch.tensor([[0.,1.]]*9)
            bounds[6:8]=torch.tensor([-.3,.3])
            warm=bounds.mean(1);warm[6:8]=torch.tensor([.2,-.1])
            data=dict(dof_pos=torch.zeros(2,2),des_dof_pos=torch.zeros(2,2),time=torch.arange(2))
            opt=module.CMAESOptimizer(bounds,4,d,['left','right'],1,data,'cpu',
                groups={'pair':[0,1]},active_groups=['pair'],warm_start=warm,
                fixed_bias=0.,segment_edges_hz=())
            self.assertEqual(opt.active_idx.tolist(),[0,1,2,3,4,5,8])
            for _ in range(3):
                opt._reset_population()
                self.assertTrue(torch.equal(opt.sim_params[:,6:8],torch.zeros(4,2)))
            self.assertTrue(torch.equal(opt._params_to_sim_params(opt._sub_to_full(opt.optimizer._mean))[6:8],torch.zeros(2)))
            opt.close()

if __name__=='__main__':unittest.main()
