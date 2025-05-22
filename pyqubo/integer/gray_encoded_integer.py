from pyqubo import Binary
from pyqubo.integer.integer import Integer, IntegerWithPenalty
from pyqubo.array import Array
from cpp_pyqubo import SubH

class GrayEncInteger(Integer):
    """Gray encoded integer
    """
    
    def __init__(self, label, value_range):
        lower, upper = value_range

        if not isinstance(lower, int) or not isinstance(upper, int):
            raise TypeError("lower and upper must be integers")
        if upper <= lower:
            raise ValueError("upper must be greater than lower")    
       
        self.lower_bound = lower
        self.upper_bound = upper

        max_val_to_encode = upper - lower

        if max_val_to_encode == 0:
            num_bits = 0
            numeric_expression = float(lower)
            self.gray_array = []
        else:
            num_bits = max_val_to_encode.bit_length()
            self.gray_array = Array.create(label + "_gray", shape=num_bits, vartype='BINARY')
            # Convert to gray code
            b_exprs = [None] * num_bits
            b_exprs[num_bits - 1] = self.gray_array[num_bits - 1]

            # b_i = g_i XOR b_{i+1}
            # XOR(A,B) = A + B - 2*A*B
            for i in range(num_bits - 2, -1, -1):
                g_i = self.gray_array[i]
                b_i_plus_1 = b_exprs[i + 1]
                b_exprs[i] = g_i + b_i_plus_1 - 2 * g_i * b_i_plus_1

            # Convert to binary code
            sum_of_powers_of_2 = sum(
                [2 ** i * b_exprs[i] for i in range(num_bits)]
            )
            numeric_expression = lower + sum_of_powers_of_2

        final_expression = SubH(numeric_expression, label)

        super().__init__(
            label=label,
            value_range=value_range,
            express=final_expression,
        )

        
      

       
           