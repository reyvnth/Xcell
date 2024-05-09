import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from .BiologicalContext import *
from .DimReduc import *
from .ModelTraining import *
from .ShapAnalysis import *


def main(args):
    start = time.time()
    # TODO: make these an input file because this is too much..

    expression_file = args.expression_file
    meta_file = args.meta_file
    data_label = args.data_label
    reduc_method = args.reduction_method
    proportion_var_explained = args.proportion_var_explained
    num_ranks = args.num_ranks
    min_k = args.min_k
    max_k = args.max_k
    covariates = args.covariates
    target = args.target
    method = args.method
    harmonize = args.harmonize
    batch_keys = args.batch_keys

    # IMPORT DATA ========
    # TODO: input formatting -- CELLS as COLUMNS, MARKERS as ROWS
    # disease_exp = pyreadr.read_r('/Users/zhanglab_mac2/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Zhang_Lab/Research/shap/conditional_nmf/disease_fib_exp.RDS')
    # disease_exp = pyreadr.read_r(expression_file)
    # disease_fib_exp = disease_exp[None]
    # disease_fib_exp = pd.DataFrame(disease_fib_exp)
    # disease_fib_exp.to_csv("/Users/zhanglab_mac2/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Zhang_Lab/Research/shap/xcell/uc_fibroblast_data/fibrobalst_exp.csv")
    disease_fib_exp = pd.read_csv(expression_file, index_col=0)
    # meta = pyreadr.read_r('/Users/zhanglab_mac2/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Zhang_Lab/Research/shap/conditional_nmf/disease_fib_meta.RDS')
    # meta = pyreadr.read_r(meta_file)
    # meta_data = meta[None]
    # meta_data = pd.DataFrame(meta_data)
    # meta_data.to_csv("/Users/zhanglab_mac2/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Zhang_Lab/Research/shap/xcell/uc_fibroblast_data/fibrobalst_met.csv")
    meta_data = pd.read_csv(meta_file, index_col=0)

    # output path
    out_path = f"../output/results/{data_label}_{reduc_method}_{num_ranks}_{method}/"
    os.makedirs(out_path, exist_ok=True)

    # save cell names and gene names
    cell_names = disease_fib_exp.columns
    gene_names = disease_fib_exp.index

    # TODO: Also add the NAM portion!!!!!!
    # CALL DIMENSIONALITY REDUCTION METHOD ========
    # W, H = nonnegativeMatrixFactorization(disease_fib_exp, numberOfComponents=num_ranks, min_k=min_k, max_k=max_k)
    """W, H = reduceDim("nmf",
                     {"X": disease_fib_exp,
                      "numberOfComponents": num_ranks,
                      "min_k": min_k,
                      "max_k": max_k}
                      )
    W = pd.DataFrame(W)
    W.set_index(disease_fib_exp.index, inplace=True)
    print("\t", W.index)
    print("Shape of W matrix: ", W.shape)
    print("Shape of H matrix: ", H.shape)"""

    # compareModels(disease_fib_exp, k_values=[4, 6, 8], outcome=meta_data[target], outpath=out_path)
    # sys.exit()

    if reduc_method == "nmf":
        W, H = nonnegativeMatrixFactorization(
            disease_fib_exp,
            numberOfComponents=num_ranks,
            min_k=min_k,
            max_k=max_k,
            outpath=out_path,
        )
    elif reduc_method == "pca":
        H = principalComponentAnalysis(
            disease_fib_exp.T, proportion_var_explained, outpath=out_path
        )

    W = pd.DataFrame(W, index=gene_names)
    H = pd.DataFrame(H, columns=cell_names).T
    # TRAIN MODEL ========
    # y_variable = "disease"
    y_variable = target
    # covariates = ['nUMI', 'sample']
    batch_keys = ["sample"]
    # save the y variable labels because apparently they are lost globally after label encoding
    y_labels = meta_data[y_variable]
    # print(y_labels)
    rf_model, X_with_covariates = train_model(
        H,
        meta_data,
        covariates,
        y_variable,
        method=method,
        harmonize=False,
        outpath=out_path,
    )  # , batch_keys=batch_keys)
    # print(rf_model)

    # SHAP COMPUTATION ========
    shap_values = runSHAP(rf_model, X_with_covariates, outpath=out_path)
    # print(shap_values[0][0])
    # name the SHAP cplumns
    shap_values_df = pd.DataFrame(
        shap_values[1],
        columns=[f"{col}_shap" for col in X_with_covariates.columns],
        index=cell_names,
    )
    meta_data[target] = y_labels
    boxplotExploration(shap_values_df, meta_data, target, outpath=out_path)
    # print(shap_values_df.columns)

    # TODO: STATISTICAL SIGNIFICANCE ANALYSIS: this section is crucial for selecting the important feature
    # STATISTICAL SIGNIFICANCE ========

    # DETERMINE BIOLOGICAL SIGNIFICANCE (IMPORTANT GENES) ========
    # NOTE : here i am dropping covariate columns (do not contain "dim") from the shap value matrix to run the geneImportance function
    shap_values_df_dims_only = shap_values_df.drop(
        list(shap_values_df.filter(regex="^(?!.*dim)")), axis=1, inplace=False
    )
    gene_importance_mat = geneImportance(
        W,
        disease_fib_exp.T,
        shap_values_df_dims_only,
        meta_data,
        y_variable,
        y_labels,
        outpath=out_path,
    )

    end = time.time()
    elapsed = end - start
    # minutes, seconds = divmod(elapsed, 60)
    # print(f'Elapsed time: {int(minutes)} minutes {seconds: .2f} seconds.')
    print(f"Elapsed time: {elapsed}.")


if __name__ == "__main__":
    # ARGUMENT PARSER =========
    parser = argparse.ArgumentParser(description="XCell input parameters.")
    # NOTE: add help page
    # input files
    parser.add_argument(
        "--expression-file",
        dest="expression_file",
        type=str,
        help="Path to expression data file.",
    )
    parser.add_argument(
        "--meta-file", dest="meta_file", type=str, help="Path to meta data file."
    )
    parser.add_argument(
        "--data",
        dest="data_label",
        type=str,
        help="Label for data (to be used when labeling output directory)",
    )
    # dimensionality reduction parameters
    parser.add_argument(
        "--reduction-method",
        dest="reduction_method",
        type=str,
        default="nmf",
        help="Dimensionality reduction method: 'nmf' or 'pca'.",
    )
    parser.add_argument(
        "--proportion-var-explained",
        dest="proportion_var_explained",
        type=float,
        default=0.95,
        help="Desired proportion of variance explained if PCA is selected.",
    )
    parser.add_argument(
        "--number-ranks",
        dest="num_ranks",
        type=int,
        default=-1,
        help="Number of ranks for NMF (default: -1).",
    )
    parser.add_argument(
        "--minimum-k",
        dest="min_k",
        type=int,
        default=2,
        help="Minimum k value for selecting optimal number of ranks if NMF is selected (default: 2).",
    )
    parser.add_argument(
        "--maximum-k",
        dest="max_k",
        type=int,
        default=7,
        help="Maximum k value for selecting optimal number of ranks if NMF is selected (default: 7).",
    )
    # classification model parameters
    parser.add_argument(
        "--covariates",
        dest="covariates",
        nargs="+",
        type=str,
        help="Covariates to include in classification model.",
    )
    parser.add_argument(
        "--target-variable",
        dest="target",
        type=str,
        help="Name of the target/outcome column in the meta data table.",
    )
    parser.add_argument(
        "--classification-method",
        dest="method",
        type=str,
        default="rf",
        help="Classification model to train - rf for Random Forest (default), xgb for XGBoost.",
    )
    parser.add_argument(
        "--harmonize", dest="harmonize", action="store_true", help="Run harmony?"
    )
    parser.add_argument(
        "--batch-keys",
        dest="batch_keys",
        nargs="+",
        type=str,
        help="If running harmony, please provide a list of batch keys to use. These should be names of columns in the metadata table.",
    )

    args = parser.parse_args()
    main(args)
