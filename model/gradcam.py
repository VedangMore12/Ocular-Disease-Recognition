import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from utils.preprocess import enhance_fundus_image


def generate_gradcam(model, image_path, predicted_label, output_path="gradcam_output.jpg"):
    model.eval()
    target_layer = list(model.layer4)[1].conv2  # last conv layer for ResNet18
    activations = []
    gradients = []

    def forward_hook(*hook_args):
        output = hook_args[-1]
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook_f = target_layer.register_forward_hook(forward_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    try:
        with Image.open(image_path) as source:
            img = enhance_fundus_image(source)
        device = next(model.parameters()).device
        transformed = transform(img)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("Image transform must return a torch.Tensor.")
        input_tensor = transformed.unsqueeze(0).to(device)

        output = model(input_tensor)
        if isinstance(predicted_label, torch.Tensor):
            class_idx = int(torch.argmax(predicted_label).item())
        else:
            class_idx = int(predicted_label)
        score = output[0, class_idx]
        model.zero_grad()
        score.backward()

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, weight in enumerate(weights):
            cam += weight * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.size[0], img.size[1]))
        cam_range = float(cam.max() - cam.min())
        if cam_range > 0:
            cam = (cam - cam.min()) / cam_range
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        img_np = np.asarray(img, dtype=np.float32) / 255
        overlay = np.clip(heatmap * 0.4 + img_np * 0.6, 0, 1)

        plt.imsave(output_path, overlay)
    finally:
        hook_f.remove()

    return output_path
