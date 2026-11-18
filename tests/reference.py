"""Independent test oracles, not production feature builders or trained models.

The full-column oracle implements the documented historical transform sequence.
It deliberately does not call release feature helpers or selector get_support().
Donor eligibility is held fixed for this arithmetic comparison; historical
grouping/coverage policies are tested separately and are not reconciled here.
"""
import numpy as np
import pandas as pd
from scipy import sparse


def full_column_reference(adata, artifact):
    donors = pd.Index(sorted(adata.obs.donor.unique()), name="donor")
    ratios = pd.DataFrame(0.0, index=donors, columns=artifact["prop_cols"])
    expression = pd.DataFrame(0.0, index=donors, columns=artifact["expr_cols"])
    values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    for donor in donors:
        rows = adata.obs.donor.to_numpy() == donor
        for label in adata.obs.loc[rows, "ct"].unique():
            selected = rows & (adata.obs.ct.to_numpy() == label)
            ratio = f"{label}_ratio"
            if ratio in ratios.columns:
                ratios.loc[donor, ratio] = selected.sum() / rows.sum()
            means = values[selected].mean(axis=0)
            for gene, value in zip(adata.var_names, means):
                column = f"{label}_{gene}"
                if column in expression.columns:
                    expression.loc[donor, column] = value
    prop_scaled = artifact["scaler_prop"].transform(ratios)
    vt_result = artifact["vt"].transform(expression)
    skb_result = artifact["skb"].transform(vt_result)
    expr_scaled = artifact["scaler_expr"].transform(skb_result)
    names = list(artifact["prop_cols"]) + list(
        artifact["skb"].get_feature_names_out(artifact["vt"].get_feature_names_out()))
    result = pd.DataFrame(np.hstack([prop_scaled, expr_scaled]), index=donors, columns=names)
    return result.loc[:, artifact["selected_features"]]


def historical_coverage(obs, threshold=28):
    """Literal grouping/slice policy from en_model.ipynb cell 2, lines 20–31."""
    counts = obs.groupby(["donor", "age", "ct"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0).dropna(axis=0, how="any")
    frame = props.reset_index()
    frame["donor"] = frame["donor"].astype(str)
    frame["donor"] = frame.groupby("donor").cumcount().astype(str).radd(frame["donor"] + "_")
    frame.set_index("donor", inplace=True)
    return frame.loc[(frame.iloc[:, 2:] == 0).sum(axis=1) < threshold]
