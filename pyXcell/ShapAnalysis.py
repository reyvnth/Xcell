import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.backends.backend_pdf import PdfPages


def runSHAP(model, X, outpath="./results/"):
    """Call the SHAP package to get contributions of the features.

    Parameters:
        model (object): classification model
        X (dataframne): latent dimensions (and covariates)

    Returns:
        ndarray: array of SHAP values
    """

    print("Running SHAP tree explainer")
    # call the tree explainer to get the SHAP values
    shap_values = shap.TreeExplainer(model).shap_values(X)
    print("Creating SHAP summary plots")

    # Save SHAP dependence plots for each feature
    plt.clf()
    with PdfPages(f"{outpath}shap_plots.pdf") as pdf:
        shap.summary_plot(shap_values[0], X, max_display=X.shape[1], show=False)
        pdf.savefig()
        plt.close()
        for feature in pd.DataFrame(X).columns:
            shap.dependence_plot(
                feature, shap_values[0], X, display_features=X, show=False
            )
            plt.suptitle(f"SHAP Dependence Plot for {feature}")
            pdf.savefig()
            plt.close()

    return shap_values


def scoreSHAP(shap_values, low_dimensional_embeddings):
    """Calculate the SHAP-derived importance score. Multiply the low dimensional embeddings by the SHAP values.

    Parameters:
        shap_values (dataframe): dataframe of SHAP values
        low_dimensional_embeddings (dataframe): rank loadings for each cell

    Returns:
        ndarray: array of the SHAP-derived importance scores
    """

    print("Calculating SHAP-derived importance scores")
    scaled_contributions = np.multiply(
        low_dimensional_embeddings.to_numpy(), shap_values.to_numpy()
    )
    scaled_shap_scores = np.sum(scaled_contributions, axis=1)

    return scaled_shap_scores


def boxplotExploration(shap_values_df, meta, outcome, outpath="./results/"):
    """_summary_

    Args:
        shap_values_df (_type_): _description_
        meta (_type_): _description_
        outcome (_type_): _description_
        outpath (str, optional): _description_. Defaults to "./results/".
    """

    print("Generating exploratory box/violin plots")

    shap_values_df = pd.DataFrame(shap_values_df)

    print(f"\tIndices for the shap dataframe... {shap_values_df.index}")
    meta = pd.DataFrame(meta, columns=[outcome], index=shap_values_df.index)
    shap_ranks = shap_values_df.columns

    print("shap ranks: ", shap_ranks)
    # print("shap vaues df: ", shap_values_df.head())
    # print(f"outcome: {meta[outcome]}")
    # meta[outcome] = meta[outcome].astype('category')
    shap_values_df_meta = shap_values_df.copy()
    shap_values_df_meta[outcome] = meta[outcome]
    print(shap_values_df_meta.head())
    fig, axes = plt.subplots(
        nrows=len(shap_ranks), ncols=1, figsize=(10, 6 * len(shap_ranks))
    )

    for i, shap_rank in enumerate(shap_ranks):
        # print(f"{i} - {shap_rank}")
        # print("\t", shap_values_df[shap_rank])
        # Use sns.violinplot instead of sns.boxplot
        ax = axes[i]
        # data = [shap_values_df[shap_rank][meta[outcome] == label].values for label in meta[outcome].cat.categories]

        # ax = plt.subplot(len(shap_ranks), 2, i + 1)
        # sns.violinplot(x=meta[outcome], y=shap_rank, data=shap_values_df, inner="quartile", ax=ax)
        # ax.violinplot(data, showmedians=True)
        sns.violinplot(x=outcome, y=shap_rank, data=shap_values_df_meta, ax=ax)

        # Move legend to the upper left
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        # Set title and labels
        ax.set_title(f"{shap_rank} values for Covid vs. Control, k=8")
        ax.set_xlabel("Disease Status")
        ax.set_ylabel(shap_rank)

    fig.tight_layout()
    fig.savefig(f"{outpath}boxplots.png")
