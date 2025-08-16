# 🎵 TempoLoRA: Rhythm-Aware Finetuning for Video Diffusion

**TempoLoRA** is a lightweight finetuning framework for video diffusion models (e.g. **WAN 2.2**) that injects **temporal rhythm awareness** using **LoRA adapters** plus a **spectral regularization loss**.
The goal: teach models to *move with natural tempo* — reducing flicker, stabilizing motion, and aligning generated videos with realistic human & environmental rhythms.

---

## ✨ Features

* 🕒 **Temporal-only LoRA** → finetune just the time/attention modules; fast and VRAM-friendly.
* 🎚 **Spectral loss** → penalizes high-frequency jitter, boosts natural motion bands (0.5–6 Hz).
* 🎵 **Hidden soundtrack alignment** → models learn to carry their own rhythm.
* ⚡ **Drop-in training cell** → no architecture surgery; one Python script/Colab cell does it.
* 🛠 **WAN 2.2 ready** → built for [Wan 2.2 TI2V and A14B Diffusers pipelines](https://huggingface.co/Wan-AI).

---

## 🚀 Installation

```bash
git clone https://github.com/yourname/TempoLoRA.git
cd TempoLoRA

# (recommended) create a new conda or venv
conda create -n tempolora python=3.10
conda activate tempolora

pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors decord einops scipy
```

---

## 📂 Dataset

TempoLoRA expects a directory of short videos + prompts file:

```
wan22_data/
  ├── clip1.mp4
  ├── clip2.mp4
  └── prompts.txt
```

**prompts.txt** format:

```
clip1.mp4|a cat running across a field
clip2.mp4|a drone shot of waves crashing on rocks
```

If no data is found, the training script synthesizes a toy dataset (moving square) so you can test the loop.

---

## 🧑‍💻 Usage

Run the training script (or Colab cell):

```bash
python train_tempolora.py
```

Key outputs:

* `wan22_lora_out/wan22_temporal_lora.safetensors` → LoRA checkpoint (ComfyUI / Diffusers compatible).
* `wan22_lora_out/sample_after_lora.mp4` → quick sample video after training.

---

## ⚙️ Config (train\_tempolora.py)

You can edit parameters inside the config dataclass:

```python
MODEL_ID      = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # or A14B
NUM_FRAMES    = 49
HEIGHT, WIDTH = 576, 1024
LORA_RANK     = 16
SPEC_CUTOFF_HZ = 6.0   # max natural frequency
SPEC_WEIGHT    = 0.05  # strength of spectral loss
TV_WEIGHT      = 0.01  # temporal smoothness
MAX_STEPS      = 800   # typical quick run
```

---

## 📊 Results

Baseline WAN 2.2 vs. TempoLoRA finetuned:

* ✅ Reduced temporal flicker on faces & textures.
* ✅ Smoother walking/camera pans.
* ✅ Emergent “tempo knob” at inference: guiding frequency alters motion style.

---

## 🔬 Research Insight

Video diffusion models naturally fall into **latent rhythmic attractors** during training. TempoLoRA exposes and guides these attractors with a simple spectral loss. This is the first step toward **unifying video & audio diffusion** — teaching models to *generate with rhythm*.

---

## 🖼 Example (after 1k steps)

| Baseline WAN 2.2       | TempoLoRA               |
| ---------------------- | ----------------------- |
| ![](docs/baseline.gif) | ![](docs/tempolora.gif) |

---

## 🤝 Acknowledgements

* Built on [WAN 2.2](https://huggingface.co/Wan-AI) (Diffusers).
* Inspired by rhythm analysis in biological motion and neuroscience.
* LoRA mechanics adapted from Hugging Face + PEFT community.

---

## 📜 License

MIT License (open-source, attribution appreciated).

