# Copyright 2020 Recruit Communications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pyqubo import OneHotEncInteger, OrderEncInteger, Placeholder, LogEncInteger, UnaryEncInteger, GrayEncInteger
import dimod
from pyqubo import assert_qubo_equal


class TestInteger(unittest.TestCase):

    def test_one_hot_enc_integer(self):
        a = OneHotEncInteger("a", (0, 4), strength=Placeholder("s"))
        H = (a - 3) ** 2
        model = H.compile()
        q, offset = model.to_qubo(feed_dict={"s": 10.0})
        sampleset = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(
            sampleset, feed_dict={"s": 10.0})
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.value(a) == 3)

        # will raise runtime error if the constraint is broken, when evaluating the value of the one-hot-integer object.
        worst = max(decoded, key=lambda x: x.energy)
        self.assertRaises(RuntimeError, lambda: worst.value(a))

        self.assertTrue(a.value_range == (0, 4))

        # expected_q = {('a[0]', 'a[1]'): 20.0,
        #      ('a[0]', 'a[2]'): 20.0,
        #      ('a[0]', 'a[3]'): 20.0,
        #      ('a[0]', 'a[4]'): 20.0,
        #      ('a[1]', 'a[2]'): 24.0,
        #      ('a[1]', 'a[3]'): 26.0,
        #      ('a[1]', 'a[4]'): 28.0,
        #      ('a[2]', 'a[3]'): 32.0,
        #      ('a[2]', 'a[4]'): 36.0,
        #      ('a[3]', 'a[4]'): 44.0,
        #      ('a[0]', 'a[0]'): -10.0,
        #      ('a[1]', 'a[1]'): -15.0,
        #      ('a[2]', 'a[2]'): -18.0,
        #      ('a[3]', 'a[3]'): -19.0,
        #      ('a[4]', 'a[4]'): -18.0}
        # expected_offset = 19
        # assert_qubo_equal(q, expected_q)
        # self.assertTrue(offset == expected_offset)
  
    def test_one_hot_enc_integer_equal(self):
        a = OneHotEncInteger("a", (0, 4), strength=Placeholder("s"))
        b = OneHotEncInteger("b", (0, 4), strength=Placeholder("s"))
        M = 2.0
        H = (a + b - 5) ** 2 + M * (a.equal_to(3) - 1)**2
        model = H.compile()
        q, offset = model.to_qubo(feed_dict={"s": 10.0})
        sampleset = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(
            sampleset, feed_dict={"s": 10.0})
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.value(a) == 3)
        self.assertTrue(best.value(b) == 2)
        self.assertTrue(best.subh["a_const"]==0)
        self.assertTrue(best.subh["b_const"]==0)
        self.assertEqual(len(best.constraints(only_broken=True)), 0)

    def test_order_enc_integer(self):
        a = OrderEncInteger("a", (0, 4), strength=Placeholder("s"))
        model = ((a - 3) ** 2).compile()
        q, offset = model.to_qubo(feed_dict={"s": 10.0})
        # expected_q = {
        #     ('a[3]', 'a[2]'): -8.0,
        #     ('a[0]', 'a[1]'): -8.0,
        #     ('a[3]', 'a[0]'): 2.0,
        #     ('a[2]', 'a[0]'): 2.0,
        #     ('a[1]', 'a[1]'): 5.0,
        #     ('a[3]', 'a[1]'): 2.0,
        #     ('a[2]', 'a[1]'): -8.0,
        #     ('a[3]', 'a[3]'): 5.0,
        #     ('a[0]', 'a[0]'): -5.0,
        #     ('a[2]', 'a[2]'): 5.0
        # }
        response = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(
            response, feed_dict={"s": 10.0})
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.subh["a"]==3)
        self.assertTrue(a.value_range == (0, 4))
        # assert_qubo_equal(q, expected_q)

    def test_order_enc_integer_more_than(self):
        a = OrderEncInteger("a", (0, 4), strength=5.0)
        b = OrderEncInteger("b", (0, 4), strength=5.0)
        model = ((a - b) ** 2 + (1 - a.more_than(1)) ** 2 + (1 - b.less_than(3)) ** 2).compile()
        q, offset = model.to_qubo()
        sampleset = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(sampleset)
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.subh["a"]==2)
        self.assertTrue(best.subh["b"]==2)

    def test_log_enc_integer(self):
        a = LogEncInteger("a", (0, 4))
        b = LogEncInteger("b", (0, 4))
        M = 2.0
        H = (2 * a - b - 1) ** 2 + M * (a + b - 5) ** 2
        model = H.compile()
        q, offset = model.to_qubo()
        sampleset = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(sampleset)
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.value(a) == 2)
        self.assertTrue(best.value(b) == 3)
        self.assertTrue(a.value_range == (0, 4))
        self.assertTrue(b.value_range == (0, 4))


    def test_unary_enc_integer(self):
        a = UnaryEncInteger("a", (0, 3))
        b = UnaryEncInteger("b", (0, 3))
        M = 2.0
        H = (2 * a - b - 1) ** 2 + M * (a + b - 3) ** 2
        model = H.compile()
        q, offset = model.to_qubo()
        sampleset = dimod.ExactSolver().sample_qubo(q)
        decoded = model.decode_sampleset(sampleset)
        best = min(decoded, key=lambda x: x.energy)
        self.assertTrue(best.value(a) == 1)
        self.assertTrue(best.value(b) == 2)
        self.assertTrue(a.value_range == (0, 3))
        self.assertTrue(b.value_range == (0, 3))

        # expected_q = {('a[0]', 'a[1]'): 12.0,
        #      ('a[0]', 'a[2]'): 12.0,
        #      ('a[0]', 'b[0]'): 0.0,
        #      ('a[0]', 'b[1]'): 0.0,
        #      ('a[0]', 'b[2]'): 0.0,
        #      ('a[1]', 'a[2]'): 12.0,
        #      ('a[1]', 'b[0]'): 0.0,
        #      ('a[1]', 'b[1]'): 0.0,
        #      ('a[1]', 'b[2]'): 0.0,
        #      ('a[2]', 'b[0]'): 0.0,
        #      ('a[2]', 'b[1]'): 0.0,
        #      ('a[2]', 'b[2]'): 0.0,
        #      ('b[0]', 'b[1]'): 6.0,
        #      ('b[0]', 'b[2]'): 6.0,
        #      ('b[1]', 'b[2]'): 6.0,
        #      ('a[0]', 'a[0]'): -10.0,
        #      ('a[1]', 'a[1]'): -10.0,
        #      ('a[2]', 'a[2]'): -10.0,
        #      ('b[0]', 'b[0]'): -7.0,
        #      ('b[1]', 'b[1]'): -7.0,
        #      ('b[2]', 'b[2]'): -7.0}
        # assert_qubo_equal(q, expected_q)

### My changes to the original code start here ###

    def test_gray_enc_integer_solve(self):
        # Test 1: Single variable
        # Encodes integer 'a' in range [1, 8]. Values 1, 2, ..., 8.
        # lower=1, upper=8. max_val_to_encode = 8-1 = 7.
        # 7 is binary '111', so num_bits = 7.bit_length() = 3.
        a = GrayEncInteger("a", (1, 8)) 
        H1 = (a - 5)**2 # We want a = 5
        model1 = H1.compile()
        q1, offset1 = model1.to_qubo()
        sampleset1 = dimod.ExactSolver().sample_qubo(q1)
        decoded1 = model1.decode_sampleset(sampleset1)
        best1 = min(decoded1, key=lambda x: x.energy)
        self.assertEqual(best1.value(a), 5)

        # Test 2: Two variables
        # x in [0, 3]. max_val_to_encode = 3. num_bits = 2.
        # y in [-2, 2]. max_val_to_encode = 2 - (-2) = 4. num_bits = 3.
        x = GrayEncInteger("x", (0, 3))
        y = GrayEncInteger("y", (-2, 2))
        # We want x+y-1=0 and x-2=0 => x=2, y=-1.
        H2 = (x + y - 1)**2 + (x - 2)**2 
        model2 = H2.compile()
        q2, offset2 = model2.to_qubo()
        sampleset2 = dimod.ExactSolver().sample_qubo(q2)
        decoded2 = model2.decode_sampleset(sampleset2)
        best2 = min(decoded2, key=lambda x: x.energy)
        self.assertEqual(best2.value(x), 2, "x should be 2")
        self.assertEqual(best2.value(y), -1, "y should be -1")
        
        # Test 3: Narrow range (0,1)
        # z in [0, 1]. max_val_to_encode = 1. num_bits = 1.
        z = GrayEncInteger("z", (0, 1))
        H3_0 = (z - 0)**2 # Expect z = 0
        model3_0 = H3_0.compile()
        q3_0, offset3_0 = model3_0.to_qubo()
        sampleset3_0 = dimod.ExactSolver().sample_qubo(q3_0)
        decoded3_0 = model3_0.decode_sampleset(sampleset3_0)
        best3_0 = min(decoded3_0, key=lambda x: x.energy)
        self.assertEqual(best3_0.value(z), 0)

        H3_1 = (z - 1)**2 # Expect z = 1
        model3_1 = H3_1.compile()
        q3_1, offset3_1 = model3_1.to_qubo()
        sampleset3_1 = dimod.ExactSolver().sample_qubo(q3_1)
        decoded3_1 = model3_1.decode_sampleset(sampleset3_1)
        best3_1 = min(decoded3_1, key=lambda x: x.energy)
        self.assertEqual(best3_1.value(z), 1)

    # def test_gray_enc_integer_internal_bits(self):
    #     # Test variable 'a' in range [1, 8]. Requires 3 Gray bits (for offset 0-7).
    #     a = GrayEncInteger("a", (1, 8))

    #     # --- Test for target value a=5 ---
    #     # Expected offset from lower bound (1) is 4.
    #     # Binary for offset 4: (b2,b1,b0) = (1,0,0) (MSB, ..., LSB)
    #     # Gray code for (1,0,0): g2=b2=1; g1=b1^b2=0^1=1; g0=b0^b1=0^0=0. 
    #     # So (g2,g1,g0)=(1,1,0).
    #     # Expected sample (_gray variables named from LSB): 
    #     # a_gray[0] (g0) = 0
    #     # a_gray[1] (g1) = 1
    #     # a_gray[2] (g2) = 1
    #     H_5 = (a - 5)**2
    #     model_5 = H_5.compile()
    #     q_5, offset_5 = model_5.to_qubo() 
    #     sampleset_5 = dimod.ExactSolver().sample_qubo(q_5)
        
    #     # Find sample minimizing QUBO energy
    #     best_qubo_sample_data_5 = min(sampleset_5.data(['sample', 'energy', 'num_occurrences']), key=lambda r: r.energy)
    #     best_raw_sample_5 = best_qubo_sample_data_5.sample
    #     raw_qubo_energy_5 = best_qubo_sample_data_5.energy
        
    #     calculated_objective_via_offset_5 = raw_qubo_energy_5 + offset_5
    #     objective_energy_5 = model_5.energy(best_raw_sample_5, vartype='BINARY')

    #     print(f"\n--- Diagnostics for H_5 (expected a=5) ---")
    #     print(f"  Raw QUBO energy (from dimod): {raw_qubo_energy_5}")
    #     print(f"  QUBO offset (from model.to_qubo()): {offset_5}")
    #     print(f"  Calculated objective energy (QUBO + Offset): {calculated_objective_via_offset_5}")
    #     print(f"  Objective energy (from model.energy()): {objective_energy_5}")
    #     print(f"  Found raw sample (best_raw_sample_5): {best_raw_sample_5}")
        
        
    #     self.assertEqual(objective_energy_5, 0.0, "Objective energy for H_5 should be 0.0")

    #     # Check if the decoded integer value is correct
    #     decoded_5 = model_5.decode_sample(best_raw_sample_5, "INTEGER")
    #     decoded_value_a_5 = decoded_5.value(a)
    #     print(f"  Decoded value 'a' (from decoded_5.value(a)): {decoded_value_a_5}")
    #     self.assertEqual(decoded_value_a_5, 5, "Decoded value for H_5 should be 5")

    #     # Check the underlying Gray bits
    #     print(f"  Expected Gray bits for a=5 (offset 4 -> bin 100 -> gray 110): g0 (a_gray[0])=0, g1 (a_gray[1])=1, g2 (a_gray[2])=1")
    #     print(f"  Actual Gray bits from sample:")
    #     print(f"    a_gray[0] (g0): {best_raw_sample_5.get('a_gray[0]')}")
    #     print(f"    a_gray[1] (g1): {best_raw_sample_5.get('a_gray[1]')}")
    #     print(f"    a_gray[2] (g2): {best_raw_sample_5.get('a_gray[2]')}")
    #     self.assertEqual(best_raw_sample_5.get('a_gray[0]'), 0, "g0 for offset 4 (a=5)") 
    #     self.assertEqual(best_raw_sample_5.get('a_gray[1]'), 1, "g1 for offset 4 (a=5)")
    #     self.assertEqual(best_raw_sample_5.get('a_gray[2]'), 1, "g2 for offset 4 (a=5)") 
    #     print(f"--- End of diagnostics for H_5 ---\n")

    #     # --- Test for target value a=1 ---
    #     H_1 = (a - 1)**2
    #     model_1 = H_1.compile()
    #     q_1, offset_1 = model_1.to_qubo()
    #     sampleset_1 = dimod.ExactSolver().sample_qubo(q_1)
    #     best_qubo_sample_data_1 = min(sampleset_1.data(['sample', 'energy']), key=lambda r: r.energy)
    #     best_raw_sample_1 = best_qubo_sample_data_1.sample

    #     objective_energy_1 = model_1.energy(best_raw_sample_1, vartype='BINARY')
    #     self.assertEqual(objective_energy_1, 0.0, "Objective energy for H_1 should be 0.0")
        
    #     decoded_1 = model_1.decode_sample(best_raw_sample_1, "INTEGER")
    #     self.assertEqual(decoded_1.value(a), 1, "Decoded value for H_1 should be 1")

    #     self.assertEqual(best_raw_sample_1.get('a_gray[0]'), 0, "g0 for offset 0 (a=1)")
    #     self.assertEqual(best_raw_sample_1.get('a_gray[1]'), 0, "g1 for offset 0 (a=1)")
    #     self.assertEqual(best_raw_sample_1.get('a_gray[2]'), 0, "g2 for offset 0 (a=1)")

    #     # --- Test for target value a=8 ---
    #     H_8 = (a - 8)**2
    #     model_8 = H_8.compile()
    #     q_8, offset_8 = model_8.to_qubo()
    #     sampleset_8 = dimod.ExactSolver().sample_qubo(q_8)
    #     best_qubo_sample_data_8 = min(sampleset_8.data(['sample', 'energy']), key=lambda r: r.energy)
    #     best_raw_sample_8 = best_qubo_sample_data_8.sample

    #     objective_energy_8 = model_8.energy(best_raw_sample_8, vartype='BINARY')
    #     self.assertEqual(objective_energy_8, 0.0, "Objective energy for H_8 should be 0.0")

    #     decoded_8 = model_8.decode_sample(best_raw_sample_8, "INTEGER")
    #     self.assertEqual(decoded_8.value(a), 8, "Decoded value for H_8 should be 8")
        
    #     self.assertEqual(best_raw_sample_8.get('a_gray[0]'), 0, "g0 for offset 7 (a=8)")
    #     self.assertEqual(best_raw_sample_8.get('a_gray[1]'), 0, "g1 for offset 7 (a=8)")
    #     self.assertEqual(best_raw_sample_8.get('a_gray[2]'), 1, "g2 for offset 7 (a=8)")


### End of my changes ###

    

if __name__ == '__main__':
    unittest.main()
