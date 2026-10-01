import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self,z:float) -> float:
        return 1 / (1+np.exp(-z))

    def relu(self,z:float) -> float:
        return np.maximum(0,z)
        
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        pre_activation = np.dot(x,w) + b
        if activation == "sigmoid":
            output = self.sigmoid(pre_activation)
            return np.round(output,5)
        output = self.relu(pre_activation)
        return np.round(output,5)
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
