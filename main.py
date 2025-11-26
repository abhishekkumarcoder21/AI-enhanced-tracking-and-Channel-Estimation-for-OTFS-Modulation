#!/usr/bin/env python3
"""
otfs_ai_demo.py

OTFS + AI tracking demo (toy / pedagogical).
- Simulates a small OTFS transmitter -> delay-Doppler channel -> receiver.
- Uses a lightweight tracker (Ridge regression if sklearn present, else EWMA).
- Produces visual plots: TX DD grid, RX DD grid (noisy), baseline equalized, AI-equalized.
- Also plots MSE vs SNR curve for baseline vs AI.

Save as: otfs_ai_demo.py
Run: python otfs_ai_demo.py
"""

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sys
import os

# Try to import Ridge from sklearn; fallback to EWMA tracker if not available
use_sklearn = True
try:
    from sklearn.linear_model import Ridge
except Exception:
    use_sklearn = False
    # no stderr flooding; just inform user later

# ---------------------------
# Utility functions
# ---------------------------
def qpsk_symbols(M, N, seed=None):
    rng = np.random.default_rng(seed)
    pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return rng.choice(pts, size=(M, N))

def isfft_dd_to_tf(X):
    """Toy ISFFT: FFT along columns then IFFT along rows (simple educational mapping)."""
    return ifft(fft(X, axis=1), axis=0)

def sfft_tf_to_dd(Y):
    """Toy SFFT (approx inverse of above)"""
    return ifft(fft(Y, axis=0), axis=1)

def tf_to_time(tf_grid, cp_len=8):
    M, N = tf_grid.shape
    time_blocks = []
    for n in range(N):
        s = ifft(tf_grid[:, n])
        time_blocks.append(s)
    tx = np.concatenate(time_blocks)
    if cp_len > 0:
        tx = np.concatenate([tx[-cp_len:], tx])
    return tx

def time_to_tf(rx, M, N, cp_len=8):
    if cp_len > 0:
        rx = rx[cp_len:]
    blocks = []
    block_len = M
    for n in range(N):
        start = n * block_len
        blocks.append(fft(rx[start:start+block_len]))
    tf = np.column_stack(blocks)
    return tf

def apply_channel(tx, paths):
    """
    Apply sparse delay + Doppler channel to time-domain signal tx.
    paths: list of (complex_gain, delay_samples, doppler_norm) where doppler_norm is cycles per sample
    """
    t = np.arange(len(tx))
    rx = np.zeros_like(tx, dtype=complex)
    for (g, d, nu) in paths:
        if d > 0:
            delayed = np.concatenate([np.zeros(d, dtype=complex), tx[:-d]])
        else:
            delayed = tx
        doppler = np.exp(1j * 2 * np.pi * nu * t)
        rx += g * delayed * doppler
    return rx

def awgn(signal, snr_db):
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def simple_pilot_estimate(dd_grid, k=3):
    """Pick top-k magnitude cells as 'estimated taps' (toy heuristic)."""
    absgrid = np.abs(dd_grid)
    flat = absgrid.flatten()
    idxs = np.argsort(flat)[-k:]
    est = np.array([dd_grid[np.unravel_index(i, dd_grid.shape)] for i in idxs])
    # if fewer than k (not likely) pad with zeros
    if est.size < k:
        est = np.pad(est, (0, k-est.size), 'constant', constant_values=0)
    return est

def mse_grid(X_true, X_est):
    # Ensure same shape (use min dims)
    m = min(X_true.shape[0], X_est.shape[0])
    n = min(X_true.shape[1], X_est.shape[1])
    return np.mean(np.abs(X_true[:m, :n] - X_est[:m, :n])**2)

# ---------------------------
# Lightweight tracker classes
# ---------------------------
class EWMATracker:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None
    def fit(self, seq):
        if len(seq) > 0:
            self.prev = np.array(seq[-1])
    def predict(self, seq):
        last = np.array(seq[-1])
        if self.prev is None:
            return last
        pred = self.alpha * last + (1.0 - self.alpha) * self.prev
        self.prev = pred
        return pred

class RidgeTracker:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.fitted = False
    def fit(self, X, Y):
        # X: (n_samples, n_features) ; Y: (n_samples, n_targets)
        # If single target vector per sample, ensure 2D
        self.model.fit(X, Y)
        self.fitted = True
    def predict(self, feat):
        if not self.fitted:
            return feat
        return self.model.predict(feat.reshape(1, -1)).ravel()

# ---------------------------
# Demo parameters
# ---------------------------
np.random.seed(0)
M = 32  # delay dimension
N = 16  # doppler/time dimension
cp_len = 8
snr_demo = 15  # for the visualization snapshot

# Channel: list of (gain, delay, doppler_norm)
paths = [
    (0.9+0.0j, 0, 0.0),
    (0.6*np.exp(1j*0.3), 3, 0.001),
    (0.5*np.exp(-1j*0.2), 7, -0.002)
]

# ---------------------------
# Build one frame (TX -> Channel -> RX)
# ---------------------------
X_tx = qpsk_symbols(M, N, seed=42)         # transmitted DD grid
tf_grid = isfft_dd_to_tf(X_tx)             # DD -> TF
tx = tf_to_time(tf_grid, cp_len=cp_len)    # TF -> time waveform

# pass through channel and add noise
rx = apply_channel(tx, paths)
rx_noisy = awgn(rx, snr_demo)

# receiver: time -> TF -> DD
tf_rx = time_to_tf(rx_noisy, M, N, cp_len=cp_len)
dd_rx = sfft_tf_to_dd(tf_rx)  # received DD grid (noisy, blurred)

# baseline pilot estimation (toy)
est_taps = simple_pilot_estimate(dd_rx, k=3)

# ---------------------------
# Build a small temporal dataset (multiple frames) to train/predict with tracker
# ---------------------------
num_frames = 8
est_history = []
true_history = []

for fr in range(num_frames):
    # slightly perturb true path gains to simulate time evolution
    perturbed = []
    true_gains = []
    for (g, d, nu) in paths:
        g_p = g * (1 + 0.04 * np.random.randn())
        nu_p = nu * (1 + 0.1 * np.random.randn() * 0.01)
        perturbed.append((g_p, d, nu_p))
        true_gains.append(g_p)
    true_history.append(np.array(true_gains))

    # generate a different random DD grid per frame for realism
    Xf = qpsk_symbols(M, N, seed=fr + 1)
    tf_f = isfft_dd_to_tf(Xf)
    tx_f = tf_to_time(tf_f, cp_len=cp_len)
    rx_f = apply_channel(tx_f, perturbed)
    rx_f = awgn(rx_f, snr_demo)
    tf_rx_f = time_to_tf(rx_f, M, N, cp_len=cp_len)
    dd_rx_f = sfft_tf_to_dd(tf_rx_f)

    est_f = simple_pilot_estimate(dd_rx_f, k=3)
    est_history.append(est_f)

# Prepare training data for ridge: features = real+imag parts flattened of est_history[i]
X_feats = []
Y_targets = []
for i in range(len(est_history)-1):
    fcur = est_history[i]
    fnext = est_history[i+1]
    feat = np.concatenate([fcur.real, fcur.imag])
    targ = np.concatenate([fnext.real, fnext.imag])
    X_feats.append(feat)
    Y_targets.append(targ)
X_feats = np.array(X_feats)
Y_targets = np.array(Y_targets)

# ---------------------------
# Choose & train tracker
# ---------------------------
if use_sklearn and len(X_feats) >= 2:
    try:
        tracker = RidgeTracker(alpha=1.0)
        tracker.fit(X_feats, Y_targets)
        tracker_type = "Ridge regression (sklearn)"
    except Exception:
        tracker = EWMATracker(alpha=0.6)
        tracker.fit(est_history)
        tracker_type = "EWMA (fallback)"
else:
    tracker = EWMATracker(alpha=0.6)
    tracker.fit(est_history)
    tracker_type = "EWMA (fallback)"

# Predict next taps from the last observed estimate
last_feat = np.concatenate([est_history[-1].real, est_history[-1].imag])
if isinstance(tracker, RidgeTracker):
    pred_flat = tracker.predict(last_feat)
else:
    pred_flat = tracker.predict(est_history)

# Convert predicted flattened vector back to complex taps
if pred_flat is None:
    pred_taps = est_history[-1]
else:
    half = len(pred_flat) // 2
    pred_taps = pred_flat[:half] + 1j * pred_flat[half:half*2]

# ---------------------------
# Simple equalization demonstration (toy)
# ---------------------------
def naive_equalize(dd_rx_grid, taps):
    """Naive equalization: divide entire DD grid by sum(taps) as a crude proxy."""
    denom = np.sum(taps) if np.sum(taps) != 0 else 1.0
    return dd_rx_grid / denom

dd_before = dd_rx.copy()
dd_baseline_eq = naive_equalize(dd_before, est_history[-1])
dd_ai_eq = naive_equalize(dd_before, pred_taps)

mse_before = mse_grid(X_tx, dd_before)
mse_baseline = mse_grid(X_tx, dd_baseline_eq)
mse_ai = mse_grid(X_tx, dd_ai_eq)

print("Tracker used: ", tracker_type)
print(f"MSE (raw)    : {mse_before:.4e}")
print(f"MSE baseline : {mse_baseline:.4e}")
print(f"MSE AI       : {mse_ai:.4e}")

# ---------------------------
# Visualizations (teacher-ready)
# ---------------------------
plt.figure(figsize=(12, 8))
plt.suptitle("OTFS + AI Tracking — Demo Visuals", fontsize=16)

plt.subplot(2, 2, 1)
plt.title("Transmitted DD Grid (|X|)")
plt.imshow(np.abs(X_tx), aspect='auto', origin='lower')
plt.xlabel("Doppler index")
plt.ylabel("Delay index")
plt.colorbar(shrink=0.6)

plt.subplot(2, 2, 2)
plt.title(f"Received DD Grid (noisy), SNR={snr_demo} dB")
plt.imshow(np.abs(dd_before), aspect='auto', origin='lower')
plt.xlabel("Doppler index")
plt.ylabel("Delay index")
plt.colorbar(shrink=0.6)

plt.subplot(2, 2, 3)
plt.title(f"Baseline Equalized (MSE={mse_baseline:.2e})")
plt.imshow(np.abs(dd_baseline_eq), aspect='auto', origin='lower')
plt.xlabel("Doppler index")
plt.ylabel("Delay index")
plt.colorbar(shrink=0.6)

plt.subplot(2, 2, 4)
plt.title(f"AI-Predicted Equalized (MSE={mse_ai:.2e})")
plt.imshow(np.abs(dd_ai_eq), aspect='auto', origin='lower')
plt.xlabel("Doppler index")
plt.ylabel("Delay index")
plt.colorbar(shrink=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------------------------
# MSE vs SNR curve (toy comparison)
# ---------------------------
snr_points = [0, 5, 10, 15, 20]
mse_baseline_list = []
mse_ai_list = []

for s in snr_points:
    rx_tmp = apply_channel(tx, paths)
    rx_tmp = awgn(rx_tmp, s)
    tf_rx_tmp = time_to_tf(rx_tmp, M, N, cp_len=cp_len)
    dd_rx_tmp = sfft_tf_to_dd(tf_rx_tmp)
    mse_b = mse_grid(X_tx, naive_equalize(dd_rx_tmp, est_history[-1]))
    mse_a = mse_grid(X_tx, naive_equalize(dd_rx_tmp, pred_taps))
    mse_baseline_list.append(mse_b)
    mse_ai_list.append(mse_a)

plt.figure(figsize=(7, 4))
plt.plot(snr_points, mse_baseline_list, 'o-', label='Baseline equalize')
plt.plot(snr_points, mse_ai_list, 'o-', label='AI-predicted equalize')
plt.xlabel('SNR (dB)')
plt.ylabel('MSE')
plt.title('MSE vs SNR — Baseline vs AI (toy demo)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save figures for embedding in PPT if needed
outdir = "otfs_demo_outputs"
os.makedirs(outdir, exist_ok=True)
fig_paths = [
    (plt.gcf(), os.path.join(outdir, "mse_vs_snr.png"))
]
# We already showed figures; if user wants saved images we can save them earlier - optional
print(f"Demo finished. Output figures can be saved from the displayed windows. If you want, I can modify the script to auto-save images to '{outdir}/'.")
