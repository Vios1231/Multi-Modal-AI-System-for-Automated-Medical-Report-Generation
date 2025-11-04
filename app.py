import gradio as gr
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTModel
import subprocess
import io

# === Load Model ===
vit_model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTModel.from_pretrained(vit_model_name)

vit_model.set_attn_implementation("eager")
vit_model.config.output_attentions = True
vit_model.eval()

# === Generate Grad-CAM style attention map ===
def generate_vit_gradcam(image):
    # Preprocess image
    inputs = feature_extractor(images=image, return_tensors="pt", resize=True)
    with torch.no_grad():
        outputs = vit_model(**inputs, output_attentions=True)

    # Ambil attention dari layer terakhir
    attn = outputs.attentions[-1]  # (batch, heads, tokens, tokens)
    attn = attn.mean(1)  # rata-rata semua heads
    attn_map = attn[0, 0, 1:].reshape(14, 14).detach().numpy()

    # Normalisasi dan resize ke ukuran asli
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = np.uint8(attn_map * 255)
    attn_map = Image.fromarray(attn_map).resize(image.size, resample=Image.BILINEAR)

    # Jadikan heatmap overlay
    heatmap = np.array(attn_map)
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap / 255.0)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    # Overlay dengan gambar asli
    overlay = Image.blend(image.convert("RGB"), Image.fromarray(colored), alpha=0.4)
    return overlay

# === Generate report with LLaMA ===
def generate_report(image, progress=gr.Progress()):
    progress(0, desc="🔍 Preprocessing image...")
    image = image.convert("RGB")

    progress(0.3, desc="🧠 Extracting visual features...")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vit_model(**inputs)
        visual_emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    progress(0.6, desc="🔥 Generating Grad-CAM heatmap...")
    heatmap_img = generate_vit_gradcam(image)

    progress(0.8, desc="💬 Generating medical report with LLaMA...")
    prompt = f"""
    You are a medical imaging expert. Based on this encoded visual feature:
    {visual_emb[:100]} ...
    Generate a concise, professional medical report describing the X-ray.
    """

    try:
        response = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        report = response.stdout.decode().strip() or "[No response received]"
    except subprocess.TimeoutExpired:
        report = "[Error generating report] LLaMA took too long to respond."
    except Exception as e:
        report = f"[Error generating report] {e}"

    progress(1.0, desc="Done!")
    return heatmap_img, report

# === Gradio Interface ===
iface = gr.Interface(
    fn=generate_report,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Textbox(label="Generated Medical Report", lines=20)
    ],
    title="🩺 Multi-Modal AI System for Medical Imaging Report Generation",
    description="Upload a chest X-ray image. The system uses ViT (Vision Transformer) + LLaMA to analyze and generate a report with a Grad-CAM-style heatmap overlay.",
)

if __name__ == "__main__":
    iface.launch()
