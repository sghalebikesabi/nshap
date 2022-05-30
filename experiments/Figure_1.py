# %%
# region imports
from operator import le
from lime import lime_tabular
from lime import lime_base
import sys
import time
from captum.attr import IntegratedGradients
import torch
import numpy as np

sys.path.insert(0, "..")
from utils import exp_kernel

from melime.melime.explainers.explainer import Explainer
from melime.melime.generators.kde_gen import KDEGen
import shapori as shap

import seaborn as sns
import matplotlib.pyplot as plt
# import shap
# endregion

# region blackbox
bound = 2
mean = 0
numfeat = 2
nruns = 2 #! set this to 10

x0 = [bound for _ in range(numfeat)]

xx_ = np.arange(-2, 2, 0.2)
xx = np.zeros((len(xx_), 2))
xx[:, 1] = 2
xx[:, 0] = xx_

vals = np.zeros((len(xx), 7, nruns, 2))


class BlackboxTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        y = torch.zeros(len(x))
        y += (x[:, 0] > 0) * 2 * x[:, 1] ** 2
        y -= (x[:, 0] <= 0) * x[:, 1] ** 2
        return y  


blackbox_torch = BlackboxTorch()


def blackbox(x):
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    y = np.zeros(len(x))
    y += (x[:, 0] > 0) * 2 * x[:, 1] ** 2
    y -= (x[:, 0] <= 0) * x[:, 1] ** 2
    return y  
# endregion

# %%
# region compute
bandwidth = 1
for run_idx in range(nruns):
    print(f"run {run_idx}")
    x = np.random.normal(size=(1000, numfeat)) + mean

    # region not shap
    # lime
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=x, mode="regression"
    )
    for idx, x_test in enumerate(xx):
        exp = explainer_lime.explain_instance(data_row=x_test, predict_fn=blackbox,)
        lime_vals = []
        for k, v in exp.local_exp[1]:
            lime_vals.append(v)
        vals[idx, 0, run_idx] = lime_vals

    for idx, x_test in enumerate(xx):
        exp = explainer_lime.explain_instance(data_row=x_test, predict_fn=blackbox,)
        lime_vals = []
        for k, v in exp.local_exp[1]:
            lime_vals.append(v)
        vals[idx, 0, run_idx] = lime_vals

    # local lime
    neighbourhood_data = x
    for idx, x_test in enumerate(xx):
        explainer_loc = lime_base.LimeBase(
            kernel_fn=lambda x: exp_kernel(x, sigma=bandwidth),
        )
        neighbourhood_data[0] = x_test
        exp = explainer_loc.explain_instance_with_data(
            neighborhood_data=neighbourhood_data,
            neighborhood_labels=blackbox(neighbourhood_data).reshape((-1, 1)),
            distances=((neighbourhood_data - x_test) ** 2).sum(1),
            label=0,
            num_features=2,
        )
        for k, v in exp[1]:
            vals[idx, 3, run_idx, k] = v

    # melime
    generator = KDEGen(verbose=True).fit(x)
    neighbourhood_data = x
    explainer = Explainer(
        model_predict=blackbox,
        generator=generator,
        local_model='Ridge',
    )
    for idx, x_test in enumerate(xx):
        explanation, counterfactual_examples = explainer.explain_instance(
            x_explain=x_test.reshape((1, -1)),
            r=bandwidth,
            n_samples=1000,
            tol_importance=1, #0.3,
            local_mini_batch_max=5,
            scale_data=False,
            weight_kernel='gaussian'
        )
        vals[idx, 4, run_idx] = explanation.model.coef_
        
    # IG
    ig = IntegratedGradients(blackbox_torch)
    attributions, approximation_error = ig.attribute(
        torch.tensor(xx, requires_grad=True),
        baselines=(0),
        method="gausslegendre",
        return_convergence_delta=True,
    )

    vals[:, 5, run_idx] = attributions.detach().numpy()

    acc_distances = np.zeros((len(xx)))
    for x_baseline in x:
        distances = np.sum((x_baseline - xx) ** 2, -1)
        dist_weights = np.exp(-distances / bandwidth ** 2)
        acc_distances += dist_weights

        for idx, x_test in enumerate(xx):
            ig = IntegratedGradients(blackbox_torch)
            attributions, approximation_error = ig.attribute(
                torch.tensor(x_test, requires_grad=True),
                baselines=(torch.tensor(x_baseline)),
                method="gausslegendre",
                return_convergence_delta=True,
            )

            vals[idx, 6, run_idx] += attributions.detach().numpy() * dist_weights[idx]
        vals[:, 6, run_idx] /= acc_distances.reshape((-1, 1))
    # endregion

    # shap
    explainer_shap = shap.KernelExplainer(blackbox, x)
    shap_values = explainer_shap.shap_values(xx, nsamples=2 ** numfeat)
    vals[:, 1, run_idx] = shap_values[0]

    for idx, x_test in enumerate(xx):
        explainer_loc_shap = shap.KernelExplainer(blackbox, x)
        shap_values_loc = explainer_loc_shap.shap_values(
            x_test,
            nsamples=2 ** numfeat,
            bandwidth=np.array([bandwidth]),
            nbrh=True,
            dist_per_coal=True,
        )
        vals[idx, 2, run_idx] = shap_values_loc[:, 0]
# endregion

# %%
# region process
import pandas as pd

method_idx_map = [
    ("LIME", 0),
    ("SHAP", 1),
    ("Neighbourhood SHAP", 2),
    ("Local Linear", 3),
    ('MeLIME', 4),
    ("Integrated Gradients", 5),
    ("Neighbourhood IG", 6),
]

results_var_0 = pd.concat(
    [
        pd.DataFrame(
            {
                "x0": xx[:, 0],
                "method": idx,
                "attribution": vals[:, m, run, 0],
                "feature": 0,
            }
        )
        for idx, m in method_idx_map
        for run in range(nruns)
    ]
    + [
        pd.DataFrame(
            {
                "x0": xx[:, 0],
                "method": idx,
                "attribution": vals[:, m, run, 1],
                "feature": 1,
            }
        )
        for idx, m in method_idx_map
        for run in range(nruns)
    ]
)


results_var_0["feature"].replace(1, 2, inplace=True)
results_var_0["feature"].replace(0, 1, inplace=True)

results_var_0.reset_index(inplace=True)
# endregion

#%%
# region plot
g = sns.relplot(
    data=results_var_0,
    x="x0",
    y="attribution",
    hue="method",
    col="feature",
    kind="line",
    legend=True,
)
g.fig.set_size_inches(7, 2)
g.fig.get_axes()[0].set(title=r"Feature-1")
g.fig.get_axes()[1].set(title=r"Feature-2")

g.fig.get_axes()[0].set(xlabel=r"$x_1$")
g.fig.get_axes()[1].set(xlabel=r"$x_1$")
g._legend.set(bbox_to_anchor=(1.15, .7)) 

g.fig.get_axes()[0].axhline(0, ls="--", c="black", linewidth=0.5)
g.fig.get_axes()[1].axhline(0, ls="--", c="black", linewidth=0.5)

plt.show()
plt.savefig("../results/figure_1.pdf", bbox_inches="tight")
# endregion


# %%
