import hashlib
import json
import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

import numpy as np
from reproduction.method import reference_direction,constrain_update,normalize_l1
from reproduction.common import mean_ci

class MethodTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(4)
        self.w=self.rng.normal(size=(7,19))
        self.u=self.rng.normal(size=(7,19))

    def test_unrestricted(self):
        np.testing.assert_allclose(constrain_update(self.u,self.w,1),self.u,atol=1e-14)

    def test_radial(self):
        q=reference_direction(self.w)
        p=constrain_update(self.u,self.w,0)
        np.testing.assert_allclose(p-q*(p*q).sum(1,keepdims=True),0,atol=1e-14)

    def test_orthogonal_scaling(self):
        p=constrain_update(self.u,self.w,0)
        for alpha in [.1,.25,.5]:
            np.testing.assert_allclose(constrain_update(self.u,self.w,alpha)-p,alpha*(self.u-p))

    def test_zero_reference(self):
        np.testing.assert_array_equal(constrain_update(self.u,np.zeros_like(self.w),0),0)
        np.testing.assert_allclose(constrain_update(self.u,np.zeros_like(self.w),.5),.5*self.u)

    def test_normalization_cancellation(self):
        scale=self.rng.uniform(.1,10,size=(7,1))
        np.testing.assert_allclose(normalize_l1(self.w),normalize_l1(scale*self.w))

    def test_reject_bad_input(self):
        for alpha in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):constrain_update(self.u,self.w,alpha)
        with self.assertRaises(ValueError):constrain_update(self.u,self.w[:2],0)


class EvidenceTests(unittest.TestCase):
    def test_data_checksums(self):
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        for name,record in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),record['sha256'])

    def test_seed_pairing_and_silence(self):
        rows=json.loads((ROOT/'data/evaluation.json').read_text())['rows']
        self.assertEqual(len(rows),67)
        for batch,n in [('MNIST main',5),('MNIST replication',5),('N-MNIST',7)]:
            sets=[]
            for condition in ['Unrestricted','Hard Freeze','Radial-only']:
                arm=[r for r in rows if r['batch']==batch and r['condition']==condition]
                ids=[r['seed'] for r in arm]
                self.assertEqual(len(ids),n);self.assertEqual(len(set(ids)),n);sets.append(set(ids))
                if batch=='N-MNIST' and condition=='Unrestricted':
                    self.assertTrue(all(r['metrics']['no_decision']==1 and r['metrics']['valid']==0 and r['metrics']['gap']==0 for r in arm))
            self.assertEqual(sets[0],sets[1]);self.assertEqual(sets[1],sets[2])

    def test_ci_is_paired_student_t(self):
        result=mean_ci([1,2,3,4,5])
        self.assertEqual(result['mean'],3)
        np.testing.assert_allclose(result['ci95'],[1.036756838522439,4.963243161477561])


if __name__=='__main__':unittest.main()
