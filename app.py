import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from models.networks import define_G

# --- 1. Model Configuration ---
class Args:
    n_class = 2
    net_G = 'base_transformer_pos_s4_dd8_dedim8'
    gpu_ids = []  # [] for CPU inference
    img_size = 256

args = Args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. Load Model & Weights ---
print(f"Loading model on {device}...")
model = define_G(args=args, gpu_ids=args.gpu_ids)

checkpoint_path = os.path.join("checkpoints", "BIT_LEVIR", "best_ckpt.pt")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_G_state_dict"])
model.to(device)
model.eval()

# --- 3. Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- 4. Prediction Function ---
def predict_change(img_a_pil, img_b_pil):
    if img_a_pil is None or img_b_pil is None:
        return None, None

    # Convert to RGB
    img_a_rgb = img_a_pil.convert("RGB")
    img_b_rgb = img_b_pil.convert("RGB")

    # Transform to tensor
    tensor_a = transform(img_a_rgb).unsqueeze(0).to(device)
    tensor_b = transform(img_b_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_a, tensor_b)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Binary mask: 0 (No change, Black), 255 (Change, White)
    mask = (pred * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode="L")

    # Create a red change overlay on Image B for clear visualization
    img_b_resized = img_b_rgb.resize((256, 256))
    img_b_np = np.array(img_b_resized)
    overlay_np = img_b_np.copy()
    overlay_np[mask == 255] = [255, 50, 50]  # Highlight changes in Red
    blended = Image.blend(img_b_resized, Image.fromarray(overlay_np), alpha=0.5)

    return mask_pil, blended

# --- 5. Gradio Web UI ---
demo = gr.Interface(
    fn=predict_change,
    inputs=[
        gr.Image(type="pil", label="Image A (Before)"),
        gr.Image(type="pil", label="Image B (After)")
    ],
    outputs=[
        gr.Image(type="pil", label="Change Detection Mask"),
        gr.Image(type="pil", label="Overlay on Image B (Red = Change)")
    ],
    title="🛰️ Remote Sensing Change Detection (BIT_CD)",
    description="Upload two satellite/aerial images (Before & After) to detect structural and landscape changes.",
    examples=[
        ["samples/A/test_2_0000_0000.png", "samples/B/test_2_0000_0000.png"]
    ] if os.path.exists("samples/A/test_2_0000_0000.png") else None
)

if __name__ == "__main__":
    demo.launch()
