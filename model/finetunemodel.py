import os

from model.train_model import train_model


def fine_tune_model(
    csv_path="full_df.csv",
    model_path="model/model_weights.pth",
    epochs=15,
    batch_size=8,
    lr=5e-5,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained weights not found at: {model_path}")

    return train_model(
        csv_path=csv_path,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        initial_weights_path=model_path,
    )


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fine_tune_model(
        csv_path=os.path.join(project_dir, "full_df.csv"),
        model_path=os.path.join(project_dir, "model", "model_weights.pth"),
    )
