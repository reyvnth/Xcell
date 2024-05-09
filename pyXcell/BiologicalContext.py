import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def geneImportance(
    gene_loadings, exp_mat, shap_mat, meta_data, y, y_variables, outpath="./results/"
):
    """Compute SHAP-informed gene importance for the top features

    Parameters:
        gene_loadings (dataframe): the rank by cell NMF loadings
        shap_mat (dataframe): dataframe of SHAP values
        meta_data (dataframe): meta data table
        y (str): name of the target label column
        y_variable (series): the target labels

    Returns:
        dataframe: the gene importance matrix (cell by key genes)
    """
    print("Gene Importance")

    important_dimensions = "dim7_shap"

    meta_data.set_index(y_variables.index, inplace=True)
    meta_data[y] = y_variables

    # print("meta_data[y]: ")
    # print("\t", meta_data[y])
    # gene_loadings = pd.DataFrame(gene_loadings)
    # get the key features (genes)
    key_feature = []
    for i in range(gene_loadings.shape[1]):
        # temp <- rownames(as.data.frame(decomp$W[,i]) %>% top_n(20))
        # temp <- names(decomp$W[,i])[which(decomp$W[,i] > as.numeric(quantile(decomp$W[,i], 0.995)))]
        # key_feature <- union(key_feature, temp)
        temp = gene_loadings.iloc[:, i].index[
            gene_loadings.iloc[:, i].values
            > np.quantile(gene_loadings.iloc[:, i].values, 0.95)
        ]
        key_feature = np.union1d(key_feature, temp)
    key_features_df = gene_loadings.loc[gene_loadings.index.isin(key_feature)]
    # print(key_features_df.head())

    # plot the key features
    """sns.clustermap(key_features_df, figsize=(20,25))
    plt.title("Heatmap of Key Features for Each Rank")
    #plt.show()
    plt.tight_layout()
    plt.savefig(f"{outpath}KeyFeaturesHeatmap_75.png")"""

    # print(f"disease labels: {meta_data[y].unique()}")
    # multiply the gene loadings from NMF decomposition (k x genes) by the matrix of SHAP values (k x cells)
    # gene_importance_mat = np.dot(shap_mat, key_features_df.T)
    # gene_importance_mat = pd.DataFrame(gene_importance_mat, columns=key_feature)
    # gene_importance_mat.set_index(meta_data.index, inplace=True)

    # TODO use instead the important rank/dimension. look at correltaion between the gene scores and the
    # would loop through these if there are more than one
    gene_importance_mat = np.outer(exp_mat, shap_mat[important_dimensions])
    gene_importance_mat = pd.DataFrame(gene_importance_mat, columns=key_feature)
    gene_importance_mat.set_index(meta_data.index, inplace=True)

    correlations = np.corrcoef(shap_mat[important_dimensions], exp_mat)

    # Extract the correlations for each column
    correlations_with_matrix = correlations[0, 1:]

    print("Correlation with each column in the matrix:", correlations_with_matrix)

    # Visualization
    plt.bar(range(len(correlations_with_matrix)), correlations_with_matrix)
    plt.xlabel("Column Index")
    plt.ylabel("Correlation")
    plt.title("Correlation with Each Column in the Matrix")
    plt.savefig(f"{outpath}corr.png")

    # print(f"Dimensions of the gene loadings df: {key_features_df.shape}")
    # print(f"Dimensions of the matrix of SHAP values: {shap_mat.shape}")
    # print(f"After multiplication, the dimensions of the final gene importance dataframe are {gene_importance_mat.shape}")
    # gene_colors = pd.DataFrame(index=gene_importance_mat.index, columns=['Gene Names'])
    # gene_colors['Gene Names'] = key_features_df
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=0.5)
    lut_disease = dict(
        zip(
            set(meta_data[y]),
            sns.cubehelix_palette(n_colors=len(set(meta_data[y])), light=0.9, dark=0.1),
        )
    )
    row_colors_disease = pd.Series(meta_data[y]).map(lut_disease)
    p = sns.clustermap(
        gene_importance_mat,
        # figsize=(20, 25),
        # cmap='vlag',
        row_colors=[row_colors_disease],
        linewidths=0,
        label="Disease",
    )
    p.fig.suptitle("Heatmap of SHAP-derived gene importance for each cell", fontsize=22)
    p.ax_heatmap.set_yticklabels([])
    # p.ax_heatmap.set_xticklabels(p.ax_heatmap.get_xmajorticklabels(), fontsize = 10)
    # for label in meta_data[y].unique():
    #    p.ax_col_dendrogram.bar(0, 0, color=lut_disease[label], label=label, linewidth=0)
    # l2 = p.ax_col_dendrogram.legend(title='Disease', loc="upper right", bbox_to_anchor=(1, 0.90), ncol=2, bbox_transform=plt.gcf().transFigure)
    hand = [Patch(facecolor=lut_disease[name]) for name in lut_disease]
    plt.legend(
        hand,
        lut_disease,
        title="Disease",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
        fontsize=20,
    )
    ax1 = p.ax_heatmap
    ax1.set_xlabel("Key Genes", fontsize=20)
    ax1.set_ylabel("Cells", fontsize=20)
    ax1.tick_params(right=False)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.savefig(f"{outpath}GeneImportanceHeatmap_dim7.png")

    return gene_importance_mat
