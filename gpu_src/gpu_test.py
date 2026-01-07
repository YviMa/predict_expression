import cudf
import cuml
import cupy as cp
from cuml.linear_model import Lasso

# 1. Test cupy
print("Testing CuPy...")
a = cp.eye(5)
print("CuPy OK")

# 2. Test cudf
print("Testing cuDF...")
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print("cuDF OK")

# 3. Test cuml
print("Testing cuML...")
X_test = cp.random.rand(10, 10).astype(cp.float32)
y_test = cp.random.rand(10).astype(cp.float32)
model = Lasso()
model.fit(X_test, y_test)
print("cuML OK - ALL SYSTEMS GO")