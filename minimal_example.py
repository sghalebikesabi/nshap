
import numpy as np
import shap

# generate data
num_imputations = 1000
x = np.random.uniform(low=-3, high=-3, size=((num_imputations, 2)))

# specify black-box
def blackbox (x) :
  return x[:, 0]+ x[:, 1]

# compute Neighbourhood SHAP values at ×[0] by passing data weights

# --- specify bandwidth
bandwidth = 1.0

# --- compute Euclidean distances
distances = np.sqrt(((x - x[0]) ** 2).sum(1))

# --- use exponential kernel of distance to compute weights 
def exponential_kernel(distances) :
  return np.exp(-(distances ** 2) / bandwidth)

unnormalised_weights = exponential_kernel(distances)
normalised_weights = unnormalised_weights / unnormalised_weights.sum()

# --- compute Neighbourhood SHAP based on the KernelSHAP implementation
explainer = shap.KernelExplainer(blackbox, x)
explainer.data.weights = normalised_weights

# --- fnull (average of blackbox over all data points) has to be computed again for weighted reference
explainer.fnull = np.array((np.sum((blackbox(x) .T * normalised_weights).T, 0)))
shap_values = explainer.shap_values(x[0])
