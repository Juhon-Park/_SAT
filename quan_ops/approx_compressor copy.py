import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


import numpy as np
import torch

class adder:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.Ci = C

    def xor_gate(self, A, B):
        return (A * (1 - B) + (1 - A) * B)

    def EFA(self):
        self.sum = self.xor_gate(self.Ci, self.xor_gate(self.A, self.B))
        self.cout = (self.A * self.Ci + self.A * self.B + self.B * self.Ci) % 2

        return self.sum, self.cout

    @staticmethod
    def test_all_cases():
        # Total number of cases
        total_cases = 2 ** 15

        # Iterate over all possible cases
        for case in range(total_cases):
            # Generate binary representation of the case and convert to list of 0s and 1s
            X = [int(bit) for bit in format(case, '015b')]

            # Assign values to X0, X1, ..., X14
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14 = X

            # Calculate golden
            X_concat = torch.tensor(X, dtype=torch.float32)
            golden = X_concat.sum(0)

            # Perform the operations as per the given code
            S1, C1 = adder(X0, X1, X2).EFA()
            S2, C2 = adder(X3, X4, X5).EFA()
            S3, C3 = adder(X7, X8, X9).EFA()
            S4, C4 = adder(X10, X11, X12).EFA()

            S6, C6 = adder(S1, S2, X6).EFA()
            S5, C5 = adder(C1, C2, C6).EFA()
            S8, C8 = adder(S3, S4, X13).EFA()
            S7, C7 = adder(C3, C4, C8).EFA()

            S11, C11 = adder(S6, S8, X14).EFA()
            S10, C10 = adder(S5, S7, C11).EFA()
            S9, C9 = adder(C5, C7, C10).EFA()

            # Calculate CPRS
            CPRS = S11 * 1 + S10 * 2 + S9 * 4 + C9 * 8

            # Compare CPRS with golden
            if CPRS != golden:
                print(f"Test case failed for input: {X}")
                print(f"CPRS: {CPRS}, Golden: {golden}")
                return

        print("All test cases passed")

# Run the test
adder.test_all_cases()
