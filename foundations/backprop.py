import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(w,x) + b
        y_pred = 1 / (1+np.exp(-z))
        loss = y_pred - y_true
        sigmoid_derivative = y_pred * (1 - y_pred) * x
        gradient_w = loss * sigmoid_derivative    
        gradient_b = loss * y_pred * (1 - y_pred)
        return np.round(gradient_w,5) , np.round(gradient_b,5)


        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass
