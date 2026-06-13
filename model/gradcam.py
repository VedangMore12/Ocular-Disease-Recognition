import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os

def generate_gradcam(model, image_path, predicted_label, output_path="gradcam_output.jpg"):
    model.eval()
    target_layer = list(model.layer4)[1].conv2  # last conv layer for ResNet18
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_np = np.array(img) / 255
    overlay = heatmap * 0.4 + img_np
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(255 * overlay)

    plt.imsave(output_path, overlay)
    hook_f.remove()
    hook_b.remove()

    return output_path
