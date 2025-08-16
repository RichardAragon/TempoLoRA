# === WAN 2.2 temporal-LoRA finetune with spectral regularization (single cell) ===
# Tested on PyTorch 2.4+ / CUDA 12.x with A100/4090. Set INSTALL=True on first run.
INSTALL = True

# ----------------------------- installs (safe to re-run) -----------------------------
if INSTALL:
    import sys, subprocess, pkgutil
    def _pip(args): subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + args.split())
    # Torch: comment this line if your env already has the right CUDA build
    try:
        import torch  # noqa: F401
    except Exception:
        _pip("torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    # Diffusers main (WAN 2.2 lives here), plus friends
    _pip("git+https://github.com/huggingface/diffusers.git")
    _pip("transformers accelerate safetensors decord einops")
    _pip("scipy")

# ----------------------------- imports -----------------------------
import os, math, json, time, random, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange
from safetensors.torch import save_file

from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import AutoTokenizer

# ----------------------------- config -----------------------------
@dataclass
class Config:
    # Which WAN 2.2?
    # "Wan-AI/Wan2.2-TI2V-5B-Diffusers" (720p TI2V, runs on a single high-end consumer GPU)
    # or "Wan-AI/Wan2.2-T2V-A14B-Diffusers" (MoE 14B, heavy; use offloading/FSDP if needed)
    MODEL_ID: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Data: put .mp4/.mov/.webm in DATA_DIR and a prompts.txt mapping "filename|prompt"
    DATA_DIR: str = "./wan22_data"
    PROMPTS_FILE: str = "prompts.txt"  # lines: "clip_0001.mp4|a cat running in grass"
    # If no data found, we synthesize a tiny toy dataset that still exercises the loop
    SYNTHETIC_IF_EMPTY: bool = True

    # IO
    OUTPUT_DIR: str = "./wan22_lora_out"
    LORA_FILENAME: str = "wan22_temporal_lora.safetensors"

    # Video & training shape
    FPS: int = 24
    NUM_FRAMES: int = 49       # 49 or 81 are common; keep modest for VRAM
    HEIGHT: int = 576          # multiples of 16; WAN 2.2 TI2V supports 720p @ 24fps too
    WIDTH: int = 1024

    # LoRA
    LORA_RANK: int = 16
    LORA_ALPHA: int = 16
    LORA_TARGET_PATTERNS: Tuple[str, ...] = ("temporal", "time")  # will fall back to any '*attn*' if none
    TRAIN_LINEAR_IN_ATTENTION_ONLY: bool = True  # safest

    # Spectral regularization (temporal smoothing)
    SPEC_CUTOFF_HZ: float = 6.0         # penalize energy above this temporal freq
    SPEC_WEIGHT: float = 0.05
    TV_WEIGHT: float = 0.01            # tiny total-variation penalty on traces

    # Optim
    BATCH_SIZE: int = 1
    GRAD_ACCUM: int = 1
    MAX_STEPS: int = 800
    LR: float = 1e-4
    WD: float = 0.0
    FP16: bool = False                  # set True on older GPUs if bf16 is unavailable
    BF16: bool = True                   # Hopper/Ada/Ampere: bf16 tends to be best
    GRAD_CLIP: float = 1.0
    SEED: int = 42
    LOG_EVERY: int = 20
    SAVE_EVERY: int = 200

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- tiny video dataset -----------------------------
try:
    import decord
    decord.bridge.set_bridge('torch')
except Exception as e:
    raise RuntimeError("Please ensure 'decord' installed correctly. pip install decord") from e

class VideoPromptDataset(Dataset):
    def __init__(self, data_dir: str, prompts_file: str, num_frames: int, height: int, width: int, fps: int):
        self.dir = data_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.items = []
        pfile = os.path.join(data_dir, prompts_file)
        if os.path.exists(pfile):
            with open(pfile, "r", encoding="utf-8") as f:
                for line in f:
                    if "|" in line:
                        fname, prompt = line.strip().split("|", 1)
                        path = os.path.join(data_dir, fname)
                        if os.path.exists(path):
                            self.items.append((path, prompt))
        # If empty and allowed, synthesize tiny dataset
        if not self.items and cfg.SYNTHETIC_IF_EMPTY:
            os.makedirs(data_dir, exist_ok=True)
            toy = os.path.join(data_dir, "toy.mp4")
            # synth video (moving square), 3 seconds
            T = self.num_frames
            H, W = self.height, self.width
            vid = torch.zeros(T, H, W, 3, dtype=torch.uint8)
            for t in range(T):
                sz = H // 6
                y = int((H - sz) * t / max(1, T - 1))
                x = int((W - sz) * (1 - math.cos(2 * math.pi * t / T)) * 0.25)
                vid[t, y:y+sz, x:x+sz, 0] = 255
            # write with diffusers export_to_video
            arr = (vid.numpy()).astype(np.uint8)
            export_to_video(arr, toy, fps=self.fps)
            with open(pfile, "w", encoding="utf-8") as f:
                f.write("toy.mp4|a minimal moving red square test video\n")
            self.items = [(toy, "a minimal moving red square test video")]

    def __len__(self): return len(self.items)

    def _read_video_uniform(self, path: str) -> torch.Tensor:
        vr = decord.VideoReader(path)
        total = len(vr)
        if total < self.num_frames:
            idx = np.linspace(0, total - 1, total).astype(int)
            frames = vr.get_batch(idx)  # (total, H, W, 3)
            # pad last frame
            pad = self.num_frames - total
            last = frames[-1:].repeat(pad, 1, 1, 1)
            frames = torch.cat([frames, last], dim=0)
        else:
            idx = np.linspace(0, total - 1, self.num_frames).astype(int)
            frames = vr.get_batch(idx)  # (T, H, W, 3)
        frames = frames.to(torch.uint8)
        return frames  # (T, H, W, 3) uint8

    def __getitem__(self, i):
        path, prompt = self.items[i]
        frames = self._read_video_uniform(path)  # (T, H, W, 3) uint8
        # resize & center-crop to (H,W)
        # decord returns torch tensor on CPU
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
        # simple letterbox + center crop via TF.resize keeping aspect
        T, C, H, W = frames.shape
        frames = TF.resize(frames, [cfg.height, cfg.width], antialias=True)  # (T,3,H,W) already target shape
        # scale to [-1,1] as typical VAE expects
        frames = frames * 2 - 1
        # (T,3,H,W) -> (1,3,T,H,W)
        video = frames.unsqueeze(0)
        return {"video": video, "prompt": prompt, "path": path}

dataset = VideoPromptDataset(cfg.DATA_DIR, cfg.PROMPTS_FILE, cfg.NUM_FRAMES, cfg.HEIGHT, cfg.WIDTH, cfg.FPS)
loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)

# ----------------------------- load WAN 2.2 pipeline -----------------------------
dtype = torch.bfloat16 if (cfg.BF16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else (torch.float16 if cfg.FP16 else torch.float32)

vae = AutoencoderKLWan.from_pretrained(cfg.MODEL_ID, subfolder="vae", torch_dtype=torch.float32)  # keep VAE in fp32 for stability
pipe = WanPipeline.from_pretrained(cfg.MODEL_ID, vae=vae, torch_dtype=dtype)
pipe.to(device)

# Try to locate components in a robust way
transformer = getattr(pipe, "transformer", None) or getattr(pipe, "dit", None) or getattr(pipe, "unet", None)
tokenizer = getattr(pipe, "tokenizer", None)
text_encoder = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)
scheduler = getattr(pipe, "scheduler", None)
if transformer is None or tokenizer is None or text_encoder is None or scheduler is None:
    raise RuntimeError("Could not locate WAN 2.2 components in the pipeline (transformer/tokenizer/text_encoder/scheduler). Update diffusers and check model card.")

# optional gradient checkpointing
if hasattr(transformer, "enable_gradient_checkpointing"):
    transformer.enable_gradient_checkpointing()

# ----------------------------- build temporal-only LoRA -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        # init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

def _module_is_temporal(name: str, mod: nn.Module, patterns: Tuple[str, ...]) -> bool:
    ln = name.lower()
    if any(p in ln for p in patterns):
        return True
    # heuristic fallback: any attention with 'attn' in name that lives in a block with 't' tag
    return ("attn" in ln and any(tag in ln for tag in ("t.", ".t_", "time", "temporal")))

def inject_lora_temporal(transformer: nn.Module,
                         patterns: Tuple[str, ...],
                         rank: int, alpha: int,
                         linear_only=True) -> Dict[str, LoRALinear]:
    loras: Dict[str, LoRALinear] = {}
    for name, module in list(transformer.named_modules()):
        if _module_is_temporal(name, module, patterns):
            # swap q/k/v/out or any Linear inside this module
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    full = f"{name}.{child_name}"
                    l = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, child_name, l)
                    loras[full] = l
                elif (not linear_only) and isinstance(child, nn.Conv1d):
                    # could extend to conv1d LoRA if WAN used 1D time-convs; keeping simple here
                    pass
    # If we found nothing with 'temporal', fall back to attention projections in general
    if not loras:
        for name, module in list(transformer.named_modules()):
            if "attn" in name.lower():
                for child_name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        full = f"{name}.{child_name}"
                        l = LoRALinear(child, rank=rank, alpha=alpha)
                        setattr(module, child_name, l)
                        loras[full] = l
    if not loras:
        raise RuntimeError("LoRA injection found no target Linear layers. Try widening LORA_TARGET_PATTERNS.")
    return loras

lora_layers = inject_lora_temporal(
    transformer, cfg.LORA_TARGET_PATTERNS, rank=cfg.LORA_RANK, alpha=cfg.LORA_ALPHA,
    linear_only=cfg.TRAIN_LINEAR_IN_ATTENTION_ONLY
)
print(f"[LoRA] Injected {len(lora_layers)} Linear adapters into temporal/attention modules.")

# ----------------------------- activation hooks for temporal traces -----------------------------
_trace_buffers: List[torch.Tensor] = []
_hook_handles = []

def _trace_hook_factory(T_expected: int):
    def _hook(_mod, _inp, out):
        # Normalize output to a [B,T] trace regardless of shape
        x = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(x): return
        with torch.no_grad():
            # Try to locate time axis by size match to T_expected
            dims = list(x.shape)
            bdim = 0
            t_axis = None
            for i, s in enumerate(dims):
                if i == bdim: continue
                if s == T_expected:
                    t_axis = i; break
            if t_axis is None:
                return
            # reduce all except batch & time
            reduce_axes = [i for i in range(x.dim()) if i not in (bdim, t_axis)]
            tr = x.float().mean(dim=reduce_axes)  # [B,T]
        _trace_buffers.append(tr)
    return _hook

def attach_trace_hooks(transformer: nn.Module, T: int, max_modules: int = 8):
    # Attach to the first few temporal/attn modules to limit overhead
    count = 0
    for name, module in transformer.named_modules():
        if ("temporal" in name.lower() or "time" in name.lower() or "attn" in name.lower()):
            _hook_handles.append(module.register_forward_hook(_trace_hook_factory(T)))
            count += 1
            if count >= max_modules:
                break

def clear_trace_hooks():
    for h in _hook_handles:
        try: h.remove()
        except: pass
    _hook_handles.clear()

# ----------------------------- spectral + coherence losses -----------------------------
def spectral_highfreq_energy(traces: List[torch.Tensor], fps: int, cutoff_hz: float) -> torch.Tensor:
    if not traces: return torch.tensor(0.0, device=device)
    loss = 0.0
    for tr in traces:
        # tr: [B,T]
        tr = tr - tr.mean(dim=-1, keepdim=True)
        X = torch.fft.rfft(tr, dim=-1)
        freqs = torch.fft.rfftfreq(tr.shape[-1], d=1.0 / fps).to(tr.device)  # [F]
        mask = (freqs >= cutoff_hz).float()[None, :]
        power = (X.abs() ** 2)
        loss = loss + (power * mask).mean()
    return loss / len(traces)

def temporal_tv(traces: List[torch.Tensor]) -> torch.Tensor:
    if not traces: return torch.tensor(0.0, device=device)
    loss = 0.0
    for tr in traces:
        dv = tr[:, 1:] - tr[:, :-1]
        loss = loss + (dv**2).mean()
    return loss / len(traces)

# ----------------------------- utils: text encode, vae encode, scheduler helpers -----------------------------
def encode_text(prompts: List[str]) -> torch.Tensor:
    # WAN 2.2 uses a T5-like text encoder; take last_hidden_state
    tok = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt")
    for k in tok: tok[k] = tok[k].to(device)
    with torch.no_grad():
        enc = text_encoder(**tok)
        hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
    return hidden.to(dtype)

def encode_video_to_latents(video_bchw3: torch.Tensor) -> torch.Tensor:
    # video input comes as [B, 3, T, H, W] in [-1,1]
    # AutoencoderKLWan expects [B, 3, T, H, W]; returns latents scaled by scaling_factor
    with torch.autocast(device_type="cuda", dtype=torch.float32) if device.type == "cuda" else torch.cuda.amp.autocast(enabled=False):
        latents = vae.encode(video_bchw3).latent_dist.sample()
    # use scaling_factor if present
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    return latents * sf

def noise_targets(latents: torch.Tensor, scheduler, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    if hasattr(scheduler, "add_noise"):
        noisy = scheduler.add_noise(latents, noise, timesteps)
    else:
        # Fallback: EDM-like sigma schedule if add_noise missing
        sigmas = scheduler.sigmas.to(latents.device)[timesteps.long()] if hasattr(scheduler, "sigmas") else None
        if sigmas is None:
            raise RuntimeError("Scheduler has no add_noise or sigmas; please update diffusers.")
        noisy = latents + noise * sigmas.view(-1, *([1] * (latents.ndim - 1)))
    # target
    if hasattr(scheduler, "get_velocity"):
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        pt = getattr(scheduler.config, "prediction_type", "epsilon")
        if pt in ("v_prediction", "v-prediction") and hasattr(scheduler, "sigmas"):
            sig = scheduler.sigmas[timesteps.long()].view(-1, *([1] * (latents.ndim - 1)))
            # velocity v = alpha * eps - sigma * x0  (alpha = sqrt(1 - sigma^2) approx when normalized)
            alpha = torch.sqrt(torch.clamp(1 - sig**2, min=1e-8))
            target = alpha * noise - sig * latents
        else:
            target = noise
    return noisy.to(dtype), target.to(dtype)

# ----------------------------- optimizer -----------------------------
trainable_params = [p for p in transformer.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable_params, lr=cfg.LR, weight_decay=cfg.WD)

# Mixed precision context
amp_dtype = torch.bfloat16 if (dtype == torch.bfloat16) else (torch.float16 if dtype == torch.float16 else None)
scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

# ----------------------------- training loop -----------------------------
step = 0
transformer.train()
print(f"[Train] starting… {len(dataset)} clips | target steps={cfg.MAX_STEPS} | dtype={dtype} | device={device}")

while step < cfg.MAX_STEPS:
    for batch in loader:
        if step >= cfg.MAX_STEPS: break
        video = batch["video"]  # [B, T, 3, H, W] or [B,1,3,T,H,W]?
        # normalize shape to [B, 3, T, H, W]
        if video.ndim == 5:
            video_b3thw = video.squeeze(0) if video.shape[0] == 1 and cfg.BATCH_SIZE == 1 else video
        elif video.ndim == 6:
            video_b3thw = video[:, 0]  # [B,3,T,H,W]
        else:
            raise RuntimeError("Unexpected video tensor shape.")
        if video_b3thw.shape[2] != cfg.NUM_FRAMES:
            # re-uniform sample in time if mismatch
            Tsrc = video_b3thw.shape[2]
            idx = torch.linspace(0, Tsrc - 1, cfg.NUM_FRAMES).round().long()
            video_b3thw = video_b3thw[:, :, idx]
        video_b3thw = video_b3thw.to(device, dtype=torch.float32)

        prompts = [batch["prompt"]] if isinstance(batch["prompt"], str) else batch["prompt"]
        text_hid = encode_text(prompts)  # [B, L, C] bf16

        latents = encode_video_to_latents(video_b3thw)  # fp32 latents then cast
        latents = latents.to(device, dtype=dtype)

        # diffusion timestep sampling
        if hasattr(scheduler, "config") and hasattr(scheduler.config, "num_train_timesteps"):
            t_int = torch.randint(0, int(scheduler.config.num_train_timesteps), (latents.shape[0],), device=device, dtype=torch.long)
        elif hasattr(scheduler, "timesteps") and len(scheduler.timesteps) > 0:
            t_int = torch.randint(0, len(scheduler.timesteps), (latents.shape[0],), device=device, dtype=torch.long)
        else:
            raise RuntimeError("Scheduler has no notion of training timesteps.")

        noisy, target = noise_targets(latents, scheduler, t_int)

        # attach hooks to collect temporal traces on the fly
        _trace_buffers.clear()
        clear_trace_hooks()
        attach_trace_hooks(transformer, T=cfg.NUM_FRAMES, max_modules=8)

        # forward
        with torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype and device.type == "cuda") else torch.enable_grad():
            # Wan transformer forward differs across branches; be robust:
            out = transformer(
                sample=noisy,                                  # some models use 'hidden_states' or 'sample'
                timestep=t_int,
                encoder_hidden_states=text_hid,
                return_dict=True
            ) if "Wan" in type(pipe).__name__ or hasattr(transformer, "forward") else transformer(noisy, t_int, text_hid)
            if isinstance(out, dict) and "sample" in out:
                pred = out["sample"]
            elif hasattr(out, "sample"):
                pred = out.sample
            elif isinstance(out, (tuple, list)):
                pred = out[0]
            else:
                pred = out

            loss_pred = F.mse_loss(pred.float(), target.float(), reduction="mean")

        # spectral regularization on collected traces (no autograd needed over hooks; they read outputs)
        traces = [t.to(device) for t in _trace_buffers]
        loss_spec = spectral_highfreq_energy(traces, cfg.FPS, cfg.SPEC_CUTOFF_HZ) * cfg.SPEC_WEIGHT
        loss_tv = temporal_tv(traces) * cfg.TV_WEIGHT

        loss = loss_pred + loss_spec + loss_tv

        # step
        opt.zero_grad(set_to_none=True)
        if scaler and scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP is not None and cfg.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if cfg.GRAD_CLIP is not None and cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP)
            opt.step()

        step += 1
        if step % cfg.LOG_EVERY == 0:
            print(f"[{step:05d}] loss={loss.item():.4f} (pred={loss_pred.item():.4f}, spec={loss_spec.item():.4f}, tv={loss_tv.item():.4f}) | traces={len(traces)}")

        if step % cfg.SAVE_EVERY == 0 or step == cfg.MAX_STEPS:
            # save LoRA weights
            state: Dict[str, torch.Tensor] = {}
            for name, l in lora_layers.items():
                state[f"{name}.lora_A.weight"] = l.lora_A.weight.detach().cpu().to(torch.float32)
                state[f"{name}.lora_B.weight"] = l.lora_B.weight.detach().cpu().to(torch.float32)
                state[f"{name}.alpha"] = torch.tensor(float(l.alpha))
                state[f"{name}.rank"] = torch.tensor(int(l.rank))
            meta = {
                "base_model": cfg.MODEL_ID,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "lora_alpha": cfg.LORA_ALPHA,
                "lora_rank": cfg.LORA_RANK,
                "temporal_only": True,
                "spec_cutoff_hz": cfg.SPEC_CUTOFF_HZ,
                "spec_weight": cfg.SPEC_WEIGHT,
                "tv_weight": cfg.TV_WEIGHT,
                "num_frames": cfg.NUM_FRAMES,
                "height": cfg.HEIGHT,
                "width": cfg.WIDTH,
                "fps": cfg.FPS,
            }
            out_path = os.path.join(cfg.OUTPUT_DIR, cfg.LORA_FILENAME)
            save_file(state, out_path, metadata={k: str(v) for k, v in meta.items()})
            print(f"[save] wrote LoRA => {out_path}")

clear_trace_hooks()
print("[Done]")

# ----------------------------- (optional) quick sanity sample with LoRA -----------------------------
# You can comment this block out to skip sampling.
try:
    # lightweight sanity check (few steps, low res)
    from diffusers.loaders import AttnProcsLayers
    # Load LoRA back into the transformer (we wired our own LoRA modules, so this just reuses in-place)
    print("[sample] generating a quick clip with the current pipe…")
    with torch.no_grad():
        vid = pipe(
            prompt="a cinematic shot of a playful corgi running on a beach, golden hour, soft backlight",
            height=min(512, cfg.HEIGHT),
            width=min(896, cfg.WIDTH),
            num_frames=min(33, cfg.NUM_FRAMES),
            num_inference_steps=20,
            guidance_scale=4.0,
        ).frames[0]
    export_to_video(vid, os.path.join(cfg.OUTPUT_DIR, "sample_after_lora.mp4"), fps=cfg.FPS)
    print("[sample] wrote sample_after_lora.mp4")
except Exception as e:
    print(f"[sample] skipped (non-fatal): {e}")
