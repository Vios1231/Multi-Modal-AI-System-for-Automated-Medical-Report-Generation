# 🩻 Multi-Modal AI for Automated X-Ray Report Generation

> From **pixels to paragraphs** — an end-to-end AI system that combines computer vision and language modeling to automatically generate medical-style reports from X-ray images.

---

## 🧠 Overview

This project integrates **Vision Transformer (ViT)** for visual understanding and **LLaMA (via Ollama)** for language generation.
Given a chest X-ray, the system:

1. Extracts visual features using a pretrained **ViT encoder**
2. Applies **Grad-CAM** to visualize the most relevant image regions
3. Generates a **medical-style textual report** using a **Large Language Model**
4. Allows users to **download the generated report** and view everything in a clean Gradio interface.

---

## 🚀 Features

✅ Vision Transformer (ViT) for image feature extraction
✅ Grad-CAM heatmap for explainability
✅ LLaMA (via Ollama) for report generation
✅ Interactive Gradio UI with:

* Progress bar animation
* Real-time report generation
* Downloadable report (.txt / .csv)
  ✅ Local execution (no API keys required)

---

## 🧩 System Workflow

```
X-Ray Image
    ↓
Vision Transformer (ViT)
    ↓
Grad-CAM Visualization
    ↓
LLaMA via Ollama
    ↓
Generated Medical Report
    ↓
User Download (Gradio UI)
```

---

## 💻 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/medical-xray-report-ai.git
cd medical-xray-report-ai
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Ensure Ollama is installed

Download and install [Ollama](https://ollama.ai/)
Then pull the LLaMA model:

```bash
ollama pull llama3
```

### 4️⃣ Run the app

```bash
python app.py
```

Gradio will launch a local web


## 📊 Screenshots of Demo

---

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.
It is **not a certified medical diagnostic tool** and should not be used for clinical decision-making.

---

## 🧑‍💻 Author

**Jonathan Alvios**
Data Scientist | AI Researcher | Medical AI Enthusiast
📍 Bandung, Indonesia
🔗 [LinkedIn](https://linkedin.com) — *(optional add your link)*
🔗 [GitHub](https://github.com/yourusername)

---

⭐ If you find this project interesting, consider giving it a **star** on GitHub!
