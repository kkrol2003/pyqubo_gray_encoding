from pyqubo import Binary
from pyqubo.integer.integer import Integer, IntegerWithPenalty

class GrayEncInteger(IntegerWithPenalty):
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
          pass

       
           