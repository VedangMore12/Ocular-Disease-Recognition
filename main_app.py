import os
import tempfile
import threading
import uuid
from pathlib import Path

import gradio as gr
from gradio.themes import Soft
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from model.gradcam import generate_gradcam
from model.train_model import DISEASE_CLASSES, evaluate_model, load_trained_model
from utils.preprocess import preprocess_image
from utils.recommend import get_recommendations


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "OCULAR_MODEL_PATH",
        str(PROJECT_DIR / "model" / "model_weights.pth"),
    )
)
CSV_PATH = PROJECT_DIR / "full_df.csv"

DISEASE_SUMMARIES = {
    "Normal": "The model did not identify one of the listed abnormalities.",
    "Diabetes": "The model identified retinal patterns associated with the diabetes class.",
    "Glaucoma": "The model identified retinal patterns associated with the glaucoma class.",
    "Cataract": "The model identified retinal patterns associated with the cataract class.",
    "Age-related Macular Degeneration": (
        "The model identified retinal patterns associated with the macular degeneration class."
    ),
    "Hypertension": "The model identified retinal patterns associated with the hypertension class.",
    "Pathological Myopia": (
        "The model identified retinal patterns associated with the pathological myopia class."
    ),
    "Other Diseases": "The model identified patterns assigned to the other diseases class.",
}


class OcularPredictor:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Trained model weights not found: {MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_trained_model(str(MODEL_PATH)).to(self.device)
        self.lock = threading.Lock()

    def predict(self, left_image, right_image, age, sex, diabetes, hypertension):
        image_paths = [path for path in (left_image, right_image) if path]
        if not image_paths:
            raise gr.Error("Upload at least one fundus image.")

        with self.lock:
            logits = []
            for image_path in image_paths:
                image_tensor = preprocess_image(image_path).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits.append(self.model(image_tensor).squeeze(0))

            averaged_logits = torch.stack(logits).mean(dim=0)
            probabilities = torch.softmax(averaged_logits, dim=0).cpu().numpy()
            predicted_index = int(np.argmax(probabilities))
            disease = DISEASE_CLASSES[predicted_index]
            score = float(probabilities[predicted_index])

            heatmaps = []
            for image_path in image_paths:
                heatmap_path = os.path.join(
                    tempfile.gettempdir(), f"ocular_gradcam_{uuid.uuid4().hex}.png"
                )
                heatmaps.append(
                    generate_gradcam(
                        self.model,
                        image_path,
                        predicted_index,
                        output_path=heatmap_path,
                    )
                )

        recommendation_disease = (
            "Diabetic Retinopathy" if disease == "Diabetes" else disease
        )
        recommendations = get_recommendations(
            recommendation_disease,
            "" if age is None else str(int(age)),
            sex or "",
            diabetes,
            hypertension,
        )
        if not recommendations:
            recommendations = "No specific recommendations are available for this class."

        report_path = self._create_report(
            disease,
            score,
            age,
            sex,
            recommendations,
        )
        class_scores = {
            name: float(probabilities[index])
            for index, name in enumerate(DISEASE_CLASSES)
        }
        summary = (
            f"### {disease}\n\n"
            f"Model score: **{score:.1%}**\n\n"
            f"{DISEASE_SUMMARIES[disease]}\n\n"
            "This result is for educational purposes and is not a medical diagnosis."
        )
        return (
            class_scores,
            summary,
            recommendations,
            heatmaps[0],
            heatmaps[1] if len(heatmaps) > 1 else None,
            report_path,
        )

    def evaluate(self):
        with self.lock:
            results = evaluate_model(
                self.model,
                csv_path=str(CSV_PATH),
                device=self.device,
            )

        metrics = [
            ("Accuracy", results["accuracy"]),
            ("Balanced accuracy", results["balanced_accuracy"]),
            ("Macro precision", results["macro_precision"]),
            ("Macro recall", results["macro_recall"]),
            ("Macro F1", results["macro_f1"]),
            ("Weighted F1", results["weighted_f1"]),
        ]
        metrics_markdown = (
            f"### Validation results ({results['validation_samples']} images)\n\n"
            "| Metric | Score |\n|:--|--:|\n"
            + "\n".join(f"| {name} | {value:.2%} |" for name, value in metrics)
            + "\n\n"
            "The report uses the fixed, patient-grouped 20% validation split. "
            "It is not an independent test-set result or a clinical performance claim."
        )

        per_class = results["per_class"]
        class_rows = [
            [
                name,
                f"{per_class[name]['precision']:.1%}",
                f"{per_class[name]['recall']:.1%}",
                f"{per_class[name]['f1-score']:.1%}",
                int(per_class[name]["support"]),
            ]
            for name in results["class_names"]
        ]
        class_table = pd.DataFrame(
            class_rows,
            columns=["Class", "Precision", "Recall", "F1 score", "Support"],
        )
        if int(class_table["Support"].sum()) != results["validation_samples"]:
            raise ValueError(
                "Per-class supports do not match the validation image count."
            )

        matrix = results["confusion_matrix"]
        confusion_figure = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=results["class_names"],
                y=results["class_names"],
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
                hovertemplate=(
                    "Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>"
                ),
            )
        )
        confusion_figure.update_layout(
            title="Confusion matrix",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            xaxis_tickangle=-35,
        )

        f1_figure = go.Figure(
            data=go.Bar(
                x=results["class_names"],
                y=[per_class[name]["f1-score"] for name in results["class_names"]],
                marker_color="#2878b5",
            )
        )
        f1_figure.update_layout(
            title="Per-class F1 score",
            yaxis_title="F1 score",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            xaxis_tickangle=-30,
        )
        return metrics_markdown, class_table, confusion_figure, f1_figure

    @staticmethod
    def _create_report(disease, score, age, sex, recommendations):
        file_descriptor, report_path = tempfile.mkstemp(suffix=".pdf")
        os.close(file_descriptor)

        report = canvas.Canvas(report_path, pagesize=A4)
        _, page_height = A4
        report.setFont("Helvetica-Bold", 18)
        report.drawString(48, page_height - 58, "Ocular Image Analysis")
        report.setFont("Helvetica", 12)
        report.drawString(48, page_height - 100, f"Model class: {disease}")
        report.drawString(48, page_height - 122, f"Model score: {score:.1%}")
        report.drawString(48, page_height - 144, f"Age: {age if age is not None else 'Not provided'}")
        report.drawString(48, page_height - 166, f"Sex: {sex or 'Not provided'}")
        report.setFont("Helvetica-Bold", 12)
        report.drawString(48, page_height - 210, "Recommendations")
        report.setFont("Helvetica", 10)
        for line_index, line in enumerate(recommendations.splitlines()):
            report.drawString(48, page_height - 232 - line_index * 16, line[:110])
        report.setFont("Helvetica-Oblique", 9)
        report.drawString(
            48,
            42,
            "Educational output only; not a medical diagnosis.",
        )
        report.save()
        return report_path


def build_app(predictor=None):
    predictor = predictor or OcularPredictor()

    with gr.Blocks(
        title="Ocular Disease Recognition",
        theme=Soft(),
    ) as app:
        gr.Markdown(
            "# Ocular Disease Recognition\n"
            "Upload one or both fundus images for a model prediction, or review "
            "the model's held-out validation metrics."
        )
        with gr.Tabs():
            with gr.Tab("Image analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        left_image = gr.Image(
                            type="filepath",
                            image_mode="RGB",
                            label="Left eye image",
                        )
                        right_image = gr.Image(
                            type="filepath",
                            image_mode="RGB",
                            label="Right eye image (optional)",
                        )
                        age = gr.Number(label="Age (optional)", precision=0)
                        sex = gr.Dropdown(
                            ["Female", "Male"],
                            label="Sex (optional)",
                            value=None,
                        )
                        diabetes = gr.Checkbox(label="Known diabetes")
                        hypertension = gr.Checkbox(label="Known hypertension")
                        analyze = gr.Button("Analyze image", variant="primary")
                    with gr.Column(scale=2):
                        prediction = gr.Label(
                            label="Class scores",
                            num_top_classes=len(DISEASE_CLASSES),
                        )
                        summary = gr.Markdown()
                        recommendations = gr.Markdown()
                        with gr.Row():
                            left_heatmap = gr.Image(label="Left eye Grad-CAM")
                            right_heatmap = gr.Image(label="Right eye Grad-CAM")
                        report_file = gr.File(label="Download report")

                analyze.click(
                    fn=predictor.predict,
                    inputs=[
                        left_image,
                        right_image,
                        age,
                        sex,
                        diabetes,
                        hypertension,
                    ],
                    outputs=[
                        prediction,
                        summary,
                        recommendations,
                        left_heatmap,
                        right_heatmap,
                        report_file,
                    ],
                )

            with gr.Tab("Validation metrics"):
                gr.Markdown(
                    "Evaluate the loaded checkpoint on the same deterministic, "
                    "patient-grouped holdout split used during training. Evaluation "
                    "runs only when requested and may take a few minutes. Retrain "
                    "with the updated pipeline before interpreting these metrics; "
                    "an existing checkpoint may have been trained with the old preprocessing."
                )
                evaluate = gr.Button("Evaluate validation set", variant="primary")
                metrics = gr.Markdown()
                per_class = gr.Dataframe(
                    headers=["Class", "Precision", "Recall", "F1 score", "Support"],
                    label="Per-class metrics",
                    interactive=False,
                )
                with gr.Row():
                    confusion_plot = gr.Plot(label="Confusion matrix")
                    f1_plot = gr.Plot(label="Per-class F1")

                evaluate.click(
                    fn=predictor.evaluate,
                    outputs=[metrics, per_class, confusion_plot, f1_plot],
                )

        gr.Markdown(
            "> **Important:** This educational model is not a substitute for an "
            "eye-care professional. Model scores are not calibrated clinical "
            "probabilities."
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, show_error=True)
