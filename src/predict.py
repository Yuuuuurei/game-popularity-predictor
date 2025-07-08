import pandas as pd
import os
from src.utils import predict_popularity, load_model, load_binner, is_classifier_model

def predict_from_csv(
    csv_path,
    model_path="model/best_model.pkl",
    binner_path="model/popularity_binner.pkl",
    use_embedding=True,
    export_csv=True,
    return_proba=True,
    output_path="prediction_results.csv"
):
    df = pd.read_csv(csv_path)
    model = load_model(model_path)
    binner = load_binner(binner_path) if is_classifier_model(model_path) else None

    preds, labels, probas = predict_popularity(
        df, model=model, binner=binner, use_embedding=use_embedding, return_proba=return_proba
    )

    df_result = df.copy()
    df_result["Predicted"] = labels

    if probas is not None:
        # Tambahkan confidence score (probabilitas maksimum)
        confidence = probas.max(axis=1)
        df_result["Confidence"] = confidence

    if export_csv:
        base_name = os.path.basename(csv_path).replace(".csv", "")
        output_name = f"{base_name}_with_predictions.csv"
        df_result.to_csv(output_name, index=False)
        print(f"✅ Prediction results saved to: {output_name}")

    return df_result