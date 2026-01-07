import cuml
import cupy as cp

# Create a small random matrix on the GPU
X = cp.random.rand(100, 10)

# Run a quick PCA
pca = cuml.PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Successfully ran PCA on GPU!")
print(f"Result shape: {X_pca.shape}")