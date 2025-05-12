"""Execution of the classification model for prediction the reactivity."""

import numpy as np
import logging


def run_classifier(
        model: "xgboost.sklearn.XGBClassifier",
        pipeline: "pycaret.internal.pipeline.Pipeline",
        feature_df: "pd.DataFrame") -> "pd.DataFrame":
    """A function for executing the classification model.

    Parameters
    ----------
    model : xgboost.sklearn.XGBClassifier
        XGBosst binary classification model.
    pipeline : pycaret.internal.pipeline.Pipeline
        Pycarret data preprocessing pipeline.
    feature_df : pd.DataFrame
        Dataframe with all the features for each C-H site.
    
    Returns
    -------
    feature_df : pd.DataFrame
        Dataframe with the prediction results. The features are removed before returning. 
    
    """
    # Initialize blank prediction results
    predicted_probas = np.zeros(shape=(len(feature_df), 2))
    predicted_labels = [None for _ in list(feature_df.index)]

    # Try transforming the features
    try:
        X_test = pipeline.transform(feature_df)
        X_test = X_test[list(model.feature_names_in_)]
    except Exception as e:
        logging.error("Error in prediction.run_classifier(): feature preprocessing failed: %s: %s", e.__class__.__name__, e)
    else:
        # Try making predictions
        try:
            predicted_probas = model.predict_proba(X_test)
            predicted_labels = model.predict(X_test)
        except Exception as e:
            logging.error("Error in prediction.run_classifier(): inference failed: %s: %s", e.__class__.__name__, e)

    # Save and return the results
    feature_df["probability_is_reactive"] = predicted_probas[:, 1]
    feature_df["is_predicted_reactive"] = [True if x == 1 else False for x in predicted_labels]
    return feature_df[
        [
            "_substrate_smiles",
            "_site_idx",
            "probability_is_reactive",
            "is_predicted_reactive",
        ]
    ]
