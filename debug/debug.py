import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

#batch_mask = np.random.choice(6000, 100)
#pd.to_pickle(batch_mask, "BATCH_MASK.pkl")
BATCH_MASK = pd.read_pickle(base_dir + "/BATCH_MASK.pkl")

#weight_matrix_1 = 0.01 *  np.random.randn(784, 50)
#pd.to_pickle(weight_matrix_1, "WEIGHT_MATRIX_1.pkl")
WEIGHT_MATRIX_1 = pd.read_pickle(base_dir + "/WEIGHT_MATRIX_1.pkl")

#weight_matrix_2 =  0.01 * np.random.randn(50, 10)
#pd.to_pickle(weight_matrix_2, "WEIGHT_MATRIX_2.pkl")
WEIGHT_MATRIX_2 = pd.read_pickle(base_dir + "/WEIGHT_MATRIX_2.pkl")
