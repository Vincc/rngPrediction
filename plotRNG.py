from math import log
from matplotlib import pyplot as plt
import numpy as np
LFSR1 = [0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.06666667, 0.033333335, 0.033333335, 0.16666667, 0.0, 0.0, 0.0, 0.1, 0.033333335, 
0.06666667, 0.033333335, 0.0, 0.06666667, 0.13333334, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.06666667, 0.0, 0.13333334, 0.06666667, 0.0, 0.13333334, 0.06666667, 0.13333334, 0.06666667, 0.1, 0.06666667, 0.06666667, 0.06666667, 0.1, 0.16666667, 0.06666667, 0.0, 0.1, 0.1, 0.16666667, 0.06666667, 0.033333335, 0.2, 0.033333335, 0.033333335, 0.033333335, 0.1, 0.1, 0.06666667, 0.16666667, 0.06666667, 0.2, 0.16666667, 0.26666668, 0.06666667, 0.13333334, 0.06666667, 0.2, 0.13333334, 0.13333334, 0.36666667, 0.16666667, 0.0, 0.1, 0.23333333, 0.1, 0.26666668, 0.16666667, 0.2, 0.13333334, 0.26666668, 0.26666668, 0.26666668, 0.16666667, 0.2, 0.1, 0.23333333, 0.06666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.2, 0.16666667, 0.16666667, 0.1, 0.13333334, 0.2, 0.13333334, 0.13333334, 0.1, 0.06666667, 0.23333333, 0.16666667, 0.06666667, 0.033333335, 0.16666667, 0.2, 0.2, 0.0, 0.16666667, 0.2, 0.23333333, 0.23333333, 0.23333333, 0.2, 0.26666668, 0.2, 0.2, 0.23333333, 0.23333333, 0.16666667, 0.13333334, 0.33333334, 0.13333334, 0.2, 0.2, 0.16666667, 0.1, 0.33333334, 0.16666667, 0.2, 0.3, 0.26666668, 0.23333333, 0.26666668, 0.2, 0.23333333, 0.33333334, 0.26666668, 0.26666668, 0.16666667, 0.23333333, 0.33333334, 0.2, 0.1, 0.33333334, 0.3, 0.06666667, 0.26666668, 0.36666667, 0.16666667, 0.1, 0.36666667, 0.2, 0.3, 0.16666667, 0.26666668, 0.16666667, 0.3, 0.1, 0.3, 0.26666668, 0.23333333, 0.13333334, 0.2, 0.26666668, 0.16666667, 0.16666667, 0.3, 0.23333333, 0.23333333, 0.2, 0.3, 0.2, 0.1, 0.2, 0.46666667, 0.33333334, 0.16666667, 0.3, 0.2, 0.26666668, 0.36666667, 0.26666668, 0.33333334, 0.26666668, 0.4, 0.16666667, 0.33333334, 0.43333334, 0.23333333, 0.3, 0.2, 0.33333334, 0.33333334, 0.2, 0.2, 0.33333334, 0.2, 0.36666667, 0.43333334, 0.36666667, 0.23333333, 0.4, 0.33333334, 0.23333333, 0.36666667, 0.23333333, 0.53333336, 0.56666666, 0.4, 0.2, 0.16666667, 0.16666667, 0.4, 0.33333334, 0.33333334, 0.4, 0.26666668, 0.3, 0.2, 0.26666668, 0.36666667, 0.23333333, 0.23333333, 0.2, 0.26666668, 0.53333336, 0.23333333, 0.26666668, 0.26666668, 0.33333334, 0.3, 0.26666668, 0.4, 0.26666668, 0.26666668, 0.36666667, 0.23333333, 0.36666667, 0.23333333, 0.33333334, 0.4, 0.16666667, 0.36666667, 0.33333334, 0.3, 0.2, 0.23333333, 0.23333333, 0.3, 0.26666668, 0.33333334, 0.33333334, 0.33333334, 0.26666668, 0.3, 0.26666668, 0.26666668, 0.33333334, 0.3, 0.26666668, 0.33333334, 0.26666668, 0.23333333, 0.26666668, 0.33333334, 0.4, 0.23333333, 0.36666667, 0.4, 0.36666667, 0.4, 0.5, 0.36666667, 0.2, 0.33333334, 0.4, 0.3, 0.53333336, 0.3, 0.23333333, 0.5, 0.33333334, 0.4, 0.43333334, 0.43333334, 0.46666667, 0.43333334, 0.26666668, 0.16666667, 0.36666667, 0.26666668, 0.4, 0.53333336, 0.33333334, 0.4, 0.26666668, 0.33333334, 0.4, 0.4, 0.46666667, 0.3, 0.26666668, 0.3, 0.3, 0.23333333, 0.5, 0.46666667, 0.5, 0.33333334, 0.43333334, 0.46666667, 0.46666667, 0.4, 0.33333334, 0.36666667, 0.46666667, 0.3, 0.26666668, 0.26666668, 0.46666667, 0.3, 0.46666667, 0.4, 0.5, 0.4, 0.26666668, 0.36666667, 0.5, 0.4, 0.26666668, 0.36666667, 0.33333334, 0.46666667, 0.46666667, 0.46666667, 0.4, 0.43333334, 0.33333334, 0.3, 0.4, 0.4, 0.5, 0.4, 0.4, 0.36666667, 0.3, 0.4, 0.36666667, 0.36666667, 0.5, 0.46666667, 0.33333334, 0.5, 0.56666666, 0.36666667, 0.3, 0.3, 0.56666666, 0.43333334, 0.4, 0.33333334, 0.56666666, 0.53333336, 0.33333334, 0.36666667, 0.36666667, 0.26666668, 0.5, 0.5, 0.4, 0.33333334, 0.46666667, 0.33333334, 0.53333336, 0.6333333, 0.46666667, 0.43333334, 0.46666667, 0.46666667, 0.5, 0.46666667, 0.3, 0.46666667, 0.36666667, 0.36666667, 0.36666667, 0.5, 0.43333334, 0.43333334, 0.36666667, 0.43333334, 0.4, 0.4, 0.43333334, 0.53333336, 0.5, 0.36666667, 0.5, 0.33333334, 0.5, 0.56666666, 0.53333336, 0.6, 0.4, 0.46666667, 0.6666667, 0.4, 0.46666667, 0.36666667, 0.46666667, 0.43333334, 0.6333333, 0.36666667, 0.46666667, 0.3, 0.6, 0.36666667, 0.33333334, 0.36666667, 0.6333333, 0.43333334, 0.53333336, 0.36666667, 0.43333334, 0.36666667, 0.5, 0.5, 0.43333334, 0.46666667, 0.5, 0.5, 0.53333336, 0.36666667, 0.46666667, 0.5, 0.36666667, 0.3, 0.33333334, 0.4, 0.46666667, 0.53333336, 0.33333334, 0.53333336, 0.5, 0.4, 0.46666667, 0.53333336, 0.4, 0.43333334, 0.6666667, 0.43333334, 0.53333336, 0.46666667, 0.46666667, 0.56666666, 0.46666667, 0.4, 0.33333334, 0.5, 0.46666667, 0.46666667, 0.5, 0.56666666, 0.53333336, 0.46666667, 0.33333334, 0.43333334, 0.4, 0.5, 0.76666665, 0.53333336, 0.33333334, 0.23333333, 0.5, 0.56666666, 0.53333336, 0.56666666, 0.43333334, 0.46666667, 0.73333335, 0.53333336, 0.43333334, 0.26666668, 0.33333334, 0.6, 0.6666667, 0.46666667, 0.56666666]
MT1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.06666667, 0.06666667, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.06666667, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.06666667, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.06666667, 0.06666667, 0.0, 0.06666667, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.06666667, 0.0, 0.0, 0.033333335, 0.0, 0.13333334, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.06666667, 0.0, 0.06666667, 0.0, 0.0, 0.033333335, 0.033333335, 0.0, 0.06666667, 0.0, 0.0, 0.1, 0.0, 0.033333335, 0.1, 0.033333335, 0.06666667, 0.0, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.06666667, 0.0, 0.033333335, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.033333335, 0.06666667, 0.0, 0.033333335, 0.033333335, 0.1, 0.13333334, 0.0, 0.1, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.06666667, 0.033333335, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.0, 0.033333335, 0.033333335, 0.06666667, 0.033333335, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.06666667, 0.033333335, 0.0, 0.13333334, 0.13333334, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.06666667, 0.0, 0.033333335, 0.0, 0.06666667, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.1, 0.0, 0.0, 0.033333335, 0.16666667, 0.0, 0.0, 0.0, 0.06666667, 0.06666667, 0.06666667, 0.1, 0.06666667, 0.06666667, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.033333335, 0.06666667, 0.1, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.1, 0.1, 0.1, 0.13333334, 0.033333335, 0.033333335, 0.06666667, 0.1, 0.06666667, 0.1, 0.13333334, 0.0, 0.0, 0.06666667, 0.1, 0.06666667, 0.033333335, 0.033333335, 0.13333334, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.1, 0.033333335, 0.06666667, 0.1, 0.033333335, 0.06666667, 0.13333334, 0.13333334, 0.06666667, 0.06666667, 0.0, 0.033333335, 0.0, 0.0, 0.06666667, 0.16666667, 0.16666667, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.13333334, 0.033333335, 0.033333335, 0.13333334, 0.0, 0.0, 0.1, 0.0, 0.033333335, 0.033333335, 0.0, 0.1, 0.1, 0.06666667, 0.06666667, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.06666667, 0.13333334, 0.1, 0.033333335, 0.033333335, 0.1, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.13333334, 0.1, 0.06666667, 0.13333334, 0.1, 0.1, 0.06666667, 0.13333334, 0.13333334, 0.0, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.033333335, 0.13333334, 0.033333335, 0.06666667, 0.06666667, 0.033333335, 0.06666667, 0.033333335, 0.06666667, 0.1, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.1, 0.0, 0.06666667, 0.16666667, 0.033333335, 0.06666667, 0.033333335, 0.16666667, 0.033333335, 0.06666667, 0.1, 0.06666667, 0.1, 0.1, 0.1, 0.06666667, 0.0, 0.06666667, 0.13333334, 0.1, 0.06666667, 0.06666667, 0.06666667, 0.033333335, 0.13333334, 0.06666667, 0.033333335, 0.16666667, 0.06666667, 0.033333335, 0.0, 0.06666667, 0.13333334, 0.033333335, 0.13333334, 0.06666667, 0.0, 0.033333335, 0.033333335, 0.1, 0.1, 0.0, 0.033333335, 0.06666667, 0.06666667, 0.033333335, 0.033333335, 0.06666667, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.13333334, 0.06666667, 0.1, 0.0, 0.1, 0.06666667, 0.033333335, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.1, 0.1, 0.1, 0.033333335, 0.033333335, 0.06666667, 0.13333334, 0.06666667, 0.1, 0.06666667, 0.1, 0.033333335, 0.1, 0.13333334, 0.0, 0.06666667, 0.033333335, 0.06666667, 0.2, 0.033333335, 0.033333335, 0.033333335, 0.1, 0.033333335, 0.033333335, 0.06666667, 0.06666667, 0.13333334, 0.0, 0.1, 0.06666667, 0.1, 0.06666667, 0.033333335]
MT2 = [0.0, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.1, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.1, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.1, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.06666667, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.0, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.06666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 
0.033333335, 0.06666667, 0.033333335, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333335, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.033333335, 0.0, 0.0, 0.06666667, 0.06666667, 0.0, 0.1, 0.0, 0.1, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.0, 0.0, 0.1, 0.0, 0.06666667, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.1, 0.06666667, 0.0, 0.0, 0.033333335, 0.0, 0.033333335, 0.0, 0.06666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.0, 0.1, 0.033333335, 0.033333335, 0.06666667, 0.033333335, 0.06666667, 0.0, 0.06666667, 0.0, 0.1, 0.06666667, 0.033333335, 0.0, 0.033333335, 0.06666667, 0.033333335, 0.0, 0.06666667, 0.0, 0.06666667, 0.1, 0.033333335, 0.0, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.16666667, 0.033333335, 0.0, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.06666667, 0.0, 0.0, 0.0, 0.16666667, 0.13333334, 0.06666667, 0.13333334, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.033333335, 0.0, 0.06666667, 0.0, 0.0, 0.06666667, 0.033333335, 0.1, 0.0, 0.033333335, 0.06666667, 0.0, 0.06666667, 0.06666667, 0.0, 0.033333335, 0.1, 0.1, 0.1, 0.0, 0.06666667, 0.033333335, 0.0, 0.06666667, 0.0, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.1, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.13333334, 0.13333334, 0.0, 0.0, 0.033333335, 0.06666667, 0.13333334, 0.033333335, 0.06666667, 0.033333335, 0.0, 0.0, 0.1, 0.06666667, 0.033333335, 0.0, 0.033333335, 0.16666667, 0.1, 0.0, 0.0, 0.13333334, 0.06666667, 0.033333335, 0.0, 0.0, 0.0, 0.06666667, 0.06666667, 0.033333335, 0.1, 0.033333335, 0.06666667, 0.1, 0.1, 0.033333335, 0.033333335, 0.06666667, 0.033333335, 0.1, 0.06666667, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.1, 0.06666667, 0.1, 0.0, 0.033333335, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.06666667, 0.0, 0.06666667, 0.0, 0.033333335, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.033333335, 0.033333335, 0.0, 0.0, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.06666667, 0.033333335, 0.06666667, 0.13333334, 0.06666667, 0.16666667, 0.033333335, 0.06666667, 0.06666667, 0.13333334, 0.033333335, 0.1, 0.1, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.0, 0.16666667, 0.06666667, 0.033333335, 0.033333335, 0.1, 0.033333335, 0.033333335, 0.13333334, 0.06666667, 0.06666667, 0.13333334, 0.06666667, 0.033333335, 0.06666667, 0.033333335, 0.033333335, 0.13333334, 0.06666667, 0.0, 0.13333334, 0.2, 0.033333335, 0.1, 0.06666667, 0.033333335, 0.16666667, 0.0, 
0.06666667, 0.1, 0.0, 0.06666667, 0.0, 0.06666667, 0.1, 0.033333335, 0.033333335, 0.06666667, 0.1, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.13333334, 0.033333335, 0.033333335, 0.13333334, 0.0, 0.1, 0.06666667, 0.06666667, 0.13333334, 0.033333335, 0.1, 0.033333335, 0.06666667, 0.06666667, 0.033333335, 0.033333335, 0.033333335, 0.06666667, 0.16666667, 0.1, 0.06666667, 0.13333334, 0.033333335, 0.16666667, 0.0, 0.033333335, 0.13333334, 0.1, 0.13333334, 0.06666667, 0.1, 0.13333334, 0.06666667, 0.1, 0.033333335, 0.2, 0.13333334, 0.16666667, 0.0, 0.16666667, 0.16666667, 0.1, 0.1, 0.13333334, 0.0, 0.13333334, 0.06666667, 0.033333335, 0.13333334, 0.033333335, 0.13333334, 0.0, 0.06666667, 0.033333335, 0.13333334, 0.1, 0.1, 0.06666667, 0.1, 0.16666667, 0.1, 0.06666667, 0.16666667, 0.06666667, 0.033333335, 0.1]
x = [i+1 for i in range(500)]
print(len(LFSR1))


pLFSR1 = np.polyfit(np.log(x), LFSR1, 1)
f1 = lambda x: pLFSR1[1] + pLFSR1[0]*np.log(x)

ytemp1 = f1(x)

pMT1 = np.polyfit(np.log(x), MT1, 1)
f2 = lambda x: pMT1[1] + pMT1[0]*np.log(x)

ytemp2 = f2(x)

# print(ytemp)

# print(pLFSR1)


plt.plot(x, MT1)
plt.plot(x, MT2)
# plt.plot(x,ytemp1)
# plt.plot(x,ytemp2)
axes = plt.gca()
axes.set_ylim([0,1])
plt.title("Cleaned Accuracy of Predictions")
plt.ylabel("Accuracy")
plt.xlabel("Training Iteration")
plt.legend(["Linear Feedback Shift Register", "Mersenne Twister"], loc="upper left")
plt.show()