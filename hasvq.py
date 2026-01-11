#@title Experiment: HAS-VQ (Multi-Precision & Pareto Optimal)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

GLOBAL_CONFIG = {
    "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "seed": 42,
    "seq_len": 1024,
    "calib_samples": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# We compare distinct HAS-VQ configurations against baselines
HAS_VQ_EXPERIMENTS = [
    {
        "name": "HAS-VQ (Mid ~3.5b)",
        "block_size": 4,           # Balanced
        "n_centroids": 2048,
        "sparsity_ratio": 0.015,   # 1.5% Sparse
        "stability_factor": 200,
        "kmeans_iter": 15
    },
    {
        "name": "HAS-VQ (High ~5.5b)",
        "block_size": 2,           # Small blocks = High fidelity
        "n_centroids": 2048,
        "sparsity_ratio": 0.020,   # 2.0% Sparse
        "stability_factor": 200,
        "kmeans_iter": 15
    }
]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(GLOBAL_CONFIG['seed'])

# ============================================================================
# 1. MATH: ROBUST & SCALABLE K-MEANS
# ============================================================================

def calculate_statistical_samples(num_vectors, n_centroids, stability_factor):
    """
    Dynamically calculates N based on the Central Limit Theorem.
    Ensures that as we vary K (centroids), we maintain statistical rigor.
    """
    target_samples = n_centroids * stability_factor
    return min(num_vectors, target_samples)

def robust_kmeans(X, num_clusters, n_iter=10, device='cuda'):
    """
    Robust Mini-Batch K-Means with 'Dead Unit' Revival.
    Optimized for GPU throughput.
    """
    N, D = X.shape
    X = X.float()

    # Initialization
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = X[indices].clone()

    batch_size = 64000

    for i in range(n_iter):
        numerator = torch.zeros_like(centroids)
        denominator = torch.zeros(num_clusters, 1, device=device)

        perm = torch.randperm(N, device=device)

        for j in range(0, N, batch_size):
            batch_idx = perm[j:j+batch_size]
            batch = X[batch_idx]

            # E-Step
            dists = torch.cdist(batch, centroids)
            labels = torch.argmin(dists, dim=1)

            # M-Step
            one_hot = F.one_hot(labels, num_clusters).float()
            numerator += one_hot.T @ batch
            denominator += one_hot.sum(dim=0).unsqueeze(1)

        # Dead Unit Handling
        mask = denominator > 1e-6

        if (~mask).any():
            # Revival Strategy: Move dead centroids to high-error regions
            valid_centroids = centroids[mask.squeeze()]
            
            # Sample subset for speed
            subset_idx = torch.randperm(N, device=device)[:10000]
            subset = X[subset_idx]
            
            sub_dists = torch.cdist(subset, valid_centroids)
            min_dists, _ = sub_dists.min(dim=1)
            
            # Pick points with worst reconstruction
            _, high_error_idx = torch.topk(min_dists, (~mask).sum())

            centroids[~mask.squeeze()] = subset[high_error_idx]
            centroids[mask.squeeze()] = numerator[mask.squeeze()] / denominator[mask.squeeze()]
        else:
            centroids = numerator / denominator

    return centroids.to(X.dtype)

# ============================================================================
# 2. HAS-VQ
# ============================================================================

class HASVQLinear(nn.Module):
    def __init__(self, in_features, out_features, config, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = config['block_size']

        n_centroids = config['n_centroids']
        num_blocks = math.ceil((out_features * in_features) / self.block_size)

        # Body (VQ)
        self.register_buffer('scales', torch.ones((out_features, 1), dtype=torch.float16))
        self.register_buffer('codebook', torch.zeros(n_centroids, self.block_size, dtype=torch.float16))
        self.register_buffer('indices', torch.zeros(num_blocks, dtype=torch.int16))

        # Tail (Sparse)
        self.register_buffer('sparse_idx', torch.tensor([], dtype=torch.int32))
        self.register_buffer('sparse_val', torch.tensor([], dtype=torch.float16))

        if bias is not None:
            self.register_buffer('bias', bias.clone().half())
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        # 1. Dequantize Body
        w_body = self.codebook[self.indices.long()].flatten()

        # Crop padding
        target_len = self.out_features * self.in_features
        if w_body.numel() > target_len:
            w_body = w_body[:target_len]

        w = w_body.view(self.out_features, self.in_features)

        # 2. Add Sparse Residuals
        if self.sparse_idx.numel() > 0:
            w_flat = w.flatten()
            w_flat.index_add_(0, self.sparse_idx.long(), self.sparse_val)

        # 3. Scale & Project
        w = w * self.scales
        bias = self.bias if hasattr(self, 'bias') and self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)

    @staticmethod
    def from_linear(module, config, hessian_diag):
        device = module.weight.device
        W = module.weight.data.float()
        rows, cols = W.shape
        bs = config['block_size']

        # --- A. Normalization ---
        scales = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        W_norm = W / scales

        # --- B. Importance (Fisher Info) ---
        H = hessian_diag.float().clamp(min=1e-6).view(1, -1).expand(rows, cols)
        Importance = W_norm.abs() * H.sqrt()

        Imp_flat = Importance.flatten()
        W_flat = W_norm.flatten()

        # --- C. Outlier Separation ---
        k_sparse = int(W_flat.numel() * config['sparsity_ratio'])
        top_imp, top_idx = torch.topk(Imp_flat, k_sparse)

        W_body = W_flat.clone()
        W_body[top_idx] = 0.0 # Hole Punching

        # --- D. Vector Quantization ---
        pad_len = (bs - (W_body.numel() % bs)) % bs
        if pad_len > 0:
            W_body_pad = torch.cat([W_body, torch.zeros(pad_len, device=device)])
        else:
            W_body_pad = W_body

        vectors = W_body_pad.view(-1, bs)

        # Use Dynamic Sampling but Robust Checks
        n_samp = calculate_statistical_samples(
            vectors.size(0), 
            config['n_centroids'], 
            config['stability_factor']
        )
        
        if n_samp < vectors.size(0):
            train_idx = torch.randperm(vectors.size(0), device=device)[:n_samp]
        else:
            train_idx = torch.arange(vectors.size(0), device=device)

        centroids = robust_kmeans(
            vectors[train_idx],
            config['n_centroids'],
            n_iter=config['kmeans_iter'],
            device=device
        )

        # Use Safer Chunk Size for Assignment
        idx_list = []
        chunk = 50000 
        for i in range(0, vectors.shape[0], chunk):
            dist = torch.cdist(vectors[i:i+chunk], centroids)
            idx_list.append(torch.argmin(dist, dim=1))
        indices = torch.cat(idx_list)

        # --- E. Residual Feedback ---
        recon = centroids[indices].flatten()
        if pad_len > 0: recon = recon[:-pad_len]

        final_sparse_vals = W_flat[top_idx] - recon[top_idx]

        # --- F. Construction ---
        layer = HASVQLinear(cols, rows, config, module.bias)
        layer.scales.copy_(scales.half())
        layer.codebook.copy_(centroids.half())
        layer.indices.copy_(indices.to(torch.int16))

        # Sort for memory coalescing
        sorted_idx, sort_p = torch.sort(top_idx.int())
        sorted_vals = final_sparse_vals[sort_p].half()
        layer.sparse_idx = sorted_idx
        layer.sparse_val = sorted_vals

        return layer.to(device)

# ============================================================================
# 3. BASELINES & UTILS
# ============================================================================

class Int4Linear(nn.Module):
    """ Standard RTN INT4 Baseline """
    def __init__(self, in_features, out_features, bias_tensor=None):
        super().__init__()
        self.register_buffer('weight_int', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scales', torch.zeros((out_features, 1), dtype=torch.float16))
        self.bias = bias_tensor.clone().half() if bias_tensor is not None else None
    def forward(self, x):
        w = self.weight_int.float() * self.scales.float()
        return F.linear(x, w.half(), self.bias)
    @staticmethod
    def from_linear(module):
        W = module.weight.data.float()
        scale = W.abs().amax(dim=1, keepdim=True) / 7.0
        w_int = torch.round(W / scale.clamp(min=1e-5)).clamp(-7, 7).to(torch.int8)
        layer = Int4Linear(module.in_features, module.out_features, module.bias)
        layer.weight_int = w_int
        layer.scales = scale.half()
        return layer.to(module.weight.device)

def get_calib_loader(tokenizer, n_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:1000])
    enc = tokenizer(text, return_tensors="pt", max_length=100000, truncation=False)
    ids = []
    for i in range(0, enc.input_ids.size(1) - 1024, 1024):
        ids.append(enc.input_ids[:, i:i+1024])
        if len(ids) >= n_samples: break
    return ids

def compute_hessian(model, tokenizer):
    print("  > [Calibration] Estimating Fisher Information (Hessian)...")
    model.eval()
    hessian = {}
    def hook(name):
        def fn(m, inp, out):
            if isinstance(inp, tuple): inp = inp[0]
            x = inp.view(-1, inp.shape[-1]).float()
            s = x.pow(2).mean(dim=0)
            if name in hessian: hessian[name] = 0.95 * hessian[name] + 0.05 * s.detach()
            else: hessian[name] = s.detach()
        return fn
    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    samples = get_calib_loader(tokenizer, GLOBAL_CONFIG['calib_samples'])
    with torch.no_grad():
        for batch in tqdm(samples, desc="Processing Batches"): model(batch.to(GLOBAL_CONFIG['device']))
    for h in handles: h.remove()
    return hessian

def evaluate_ppl(model, tokenizer):
    model.eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    nlls = []
    for i in tqdm(range(0, enc.input_ids.size(1), 1024), desc="Evaluating PPL"):
        end = min(i + 1024, enc.input_ids.size(1))
        if end - i < 100: continue
        ids = enc.input_ids[:, i:end].to(GLOBAL_CONFIG['device'])
        trg = ids.clone(); trg[:, :-(end-i)] = -100
        with torch.no_grad(): nlls.append(model(ids, labels=trg).loss)
    return torch.exp(torch.stack(nlls).mean()).item()

def get_stats(model):
    bits_body = 0
    bits_sparse = 0
    bits_meta = 0
    total_params = 0

    for m in model.modules():
        if isinstance(m, HASVQLinear):
            # Body: Log2(n_centroids) per block
            n_centroids = m.codebook.shape[0]
            bits_per_idx = math.ceil(math.log2(n_centroids))
            b_b = m.indices.numel() * bits_per_idx

            # Tail: 32-bit index + 16-bit val
            b_t = m.sparse_idx.numel() * (32 + 16)

            # Meta: Codebook (16b) + Scales (16b)
            b_m = m.codebook.numel()*16 + m.scales.numel()*16

            bits_body += b_b
            bits_sparse += b_t
            bits_meta += b_m
            total_params += (m.out_features * m.in_features)

        elif isinstance(m, Int4Linear):
            bits_body += m.weight_int.numel()*4
            bits_meta += m.scales.numel()*16
            total_params += m.weight_int.numel()

        elif isinstance(m, nn.Linear):
            bits_body += m.weight.numel()*16
            total_params += m.weight.numel()

    total_bits = bits_body + bits_sparse + bits_meta
    return total_bits, total_params

# ============================================================================
# 4. MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(GLOBAL_CONFIG['model_id'])
    
    # Store results for final table
    results = []

    # ------------------------------------------------------------------------
    # STEP 0: CALIBRATION & BASELINES
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 0: Initialization & Baselines")
    print("="*50)
    
    # Load Model (FP16)
    model = AutoModelForCausalLM.from_pretrained(GLOBAL_CONFIG['model_id'], torch_dtype=torch.float16, device_map="cuda")
    
    # Pre-calculate Hessian (CPU offload to save VRAM for later)
    hessian_gpu = compute_hessian(model, tokenizer)
    hessian_cpu = {k: v.cpu() for k, v in hessian_gpu.items()}
    del hessian_gpu
    
    # Baseline 1: FP16
    ppl_fp16 = evaluate_ppl(model, tokenizer)
    results.append({"Method": "FP16 (Oracle)", "PPL": ppl_fp16, "BPP": 16.0})
    print(f"FP16 PPL: {ppl_fp16:.2f}")

    del model; gc.collect(); torch.cuda.empty_cache()

    # Baseline 2: INT4
    model = AutoModelForCausalLM.from_pretrained(GLOBAL_CONFIG['model_id'], torch_dtype=torch.float16, device_map="cuda")
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            p = model.get_submodule(n.rsplit('.',1)[0]) if '.' in n else model
            setattr(p, n.split('.')[-1], Int4Linear.from_linear(m))
    
    ppl_int4 = evaluate_ppl(model, tokenizer)
    bits_i4, params_i4 = get_stats(model)
    results.append({"Method": "INT4 (RTN)", "PPL": ppl_int4, "BPP": bits_i4/params_i4})
    print(f"INT4 PPL: {ppl_int4:.2f}")
    
    del model; gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # STEP 1, 2, 3: HAS-VQ EXPERIMENTS (Multi-Precision)
    # ------------------------------------------------------------------------
    
    for exp_config in HAS_VQ_EXPERIMENTS:
        print("\n" + "="*50)
        print(f"RUNNING: {exp_config['name']}")
        print(f"Config: Block={exp_config['block_size']}, Sparse={exp_config['sparsity_ratio']*100}%")
        print("="*50)

        # Reload clean model
        model = AutoModelForCausalLM.from_pretrained(GLOBAL_CONFIG['model_id'], torch_dtype=torch.float16, device_map="cuda")
        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in n and "embed" not in n]

        start_time = time.time()
        
        # Apply Compression
        for n, m in tqdm(layers, desc="Compressing"):
            # Fetch pre-calc hessian from CPU (we do use cpu on purpose not to get heavy on vram)
            h = hessian_cpu.get(n, torch.ones(m.in_features)).to(GLOBAL_CONFIG['device'])
            
            # Compress
            q = HASVQLinear.from_linear(m, exp_config, h)
            
            # Replace
            p = model.get_submodule(n.rsplit('.',1)[0]) if '.' in n else model
            setattr(p, n.split('.')[-1], q)
            
            # Cleanup
            del m, h; torch.cuda.empty_cache()

        duration = time.time() - start_time
        
        # Evaluate
        ppl = evaluate_ppl(model, tokenizer)
        bits, params = get_stats(model)
        bpp = bits / params
        
        results.append({
            "Method": exp_config['name'],
            "PPL": ppl,
            "BPP": bpp,
            "Time": duration
        })
        
        del model; gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------------
    print("\n\n" + "="*95)
    print(f"{'FINAL COMPREHENSIVE RESULT TABLE':^95}")
    print("="*95)
    print(f"Model: {GLOBAL_CONFIG['model_id']}")
    print("-" * 95)
    print(f"{'Method':<25} | {'PPL':<8} | {'BPP':<6} | {'Compr. Ratio':<12} | {'Notes'}")
    print("-" * 95)
    
    for r in results:
        ratio = 16.0 / r['BPP']
        note = ""
        if "FP16" in r['Method']: note = "Oracle"
        elif "INT4" in r['Method']: note = "Baseline"
        elif r['BPP'] < 2.5: note = "Extreme Compr."
        elif r['BPP'] < 4.5: note = "Pareto Target"
        else: note = "Higher Fidelity"
        
        print(f"{r['Method']:<25} | {r['PPL']:<8.2f} | {r['BPP']:<6.2f} | {ratio:<5.1f}x       | {note}")
    
    print("="*95)
    
    # Analysis
    best_ppl_method = min(results, key=lambda x: x['PPL'])
    best_bpp_method = min(results, key=lambda x: x['BPP'])
    
    print("\n[Analysis]")
    print(f"1. Best Accuracy: {best_ppl_method['Method']} ({best_ppl_method['PPL']:.2f} PPL)")
    print(f"2. Best Compression: {best_bpp_method['Method']} ({best_bpp_method['BPP']:.2f} bits)")
    
    # Pareto Check
    ours_mid = [r for r in results if "Mid" in r['Method']][0]
    int4 = [r for r in results if "INT4" in r['Method']][0]
    
    if ours_mid['PPL'] < int4['PPL'] and ours_mid['BPP'] < int4['BPP']:
        print(">> PARETO DOMINANCE: 'Mid' Config beats INT4 in both Accuracy and Size.")
    elif ours_mid['PPL'] < int4['PPL']:
        print(">> HIGH FIDELITY: 'Mid' Config is more accurate than INT4.")
    else:
        print(">> EFFICIENCY: 'Mid' Config is smaller than INT4.")

# ===============================================================================================
#                                FINAL COMPREHENSIVE RESULT TABLE                                
# ===============================================================================================
# Model: HuggingFaceTB/SmolLM2-1.7B-Instruct
# -----------------------------------------------------------------------------------------------
# Method                    | PPL      | BPP    | Compr. Ratio | Notes
# -----------------------------------------------------------------------------------------------
# FP16 (Oracle)             | 10.04    | 16.00  | 1.0  x       | Oracle
# INT4 (RTN)                | 20.03    | 4.71   | 3.4  x       | Baseline
# HAS-VQ (Mid ~3.5b)        | 14.23    | 4.23   | 3.8  x       | Pareto Target
# HAS-VQ (High ~5.5b)       | 10.12    | 7.03   | 2.3  x       | Higher Fidelity
# ===============================================================================================        