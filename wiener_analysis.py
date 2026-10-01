#!/usr/bin/env python3
"""
wiener_analysis.py - Step-by-step test and analysis of the Wiener filter BEFORE SSL.

Standalone: the Wiener core (identical to prefilter.py) is built in.

  python3 wiener_analysis.py \
      --calib  ~/dataset/S0_calib.wav \
      --known  ~/dataset/S1_known.wav \
      --target ~/dataset/S2_target.wav \
      --mix    ~/dataset/S3_both.wav ~/dataset/S4_cross.wav \
      --cfg    ~/uma16.cfg \
      --outdir ~/dataset/analysis_A

Steps (each prints PASS/WARN/FAIL and writes figures):
  0  Input check        : sample rate, channels, duration, clipping, dead mics
  1  Known-source profile: PSD, tonal lines, flatness, energy over time, stability, per-mic energy
  2  Spectral overlap   : how much of the target shares the known source's bins
  3  Filtering          : runs the Wiener filter on every file, saves *_filtA.wav/.raw, CPU per hop
  4  Gain behaviour     : gain map (time x freq) and mean gain vs frequency
  5  Spectrograms       : before / after / difference for each file
  6  Energy over time   : in-band energy before vs after
  7  Metrics            : known reduction (S1), target loss (S2), residual (mixes)
  8  Direction check    : GCC-PHAT delays + SRP-PHAT DOA before vs after (needs --cfg)
Final report: <outdir>/report.html (all figures + tables), metrics.json
"""
import argparse
import base64
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================ Wiener core (standalone, same as prefilter.py)
DEFAULTS = dict(frame=1024, hop=512, fmin=180.0, fmax=3600.0,
                alpha=1.5, floor=0.03, beta=0.98, full_band=False)
EPS = 1e-12


def sqrt_hann(n):
    # periodic Hann; sqrt-Hann analysis * sqrt-Hann synthesis sums to 1 at 50% overlap
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))


# ---------------------------------------------------------------- core (streaming, reusable in feeder_filter.py)
class WienerPrefilter:
    """Streaming multichannel Wiener. Feed hop x C blocks, get hop x C blocks (delay = frame - hop)."""

    def __init__(self, noise_psd, sr, n_ch, frame=1024, hop=512, fmin=180.0, fmax=3600.0,
                 alpha=1.5, floor=0.03, beta=0.98, full_band=False, **_):
        self.N, self.H, self.C = frame, hop, n_ch
        self.win = sqrt_hann(frame)[:, None]
        self.noise = np.maximum(np.asarray(noise_psd, dtype=np.float64), EPS)
        self.alpha, self.floor, self.beta = alpha, floor, beta
        f = np.fft.rfftfreq(frame, 1.0 / sr)
        self.band = np.ones_like(f, bool) if full_band else (f >= fmin) & (f <= fmax)
        self.in_buf = np.zeros((frame, n_ch))
        self.out_buf = np.zeros((frame, n_ch))
        self.prev_s2 = np.zeros(len(f))          # |S_hat|^2 of previous frame (mic-averaged)
        self.last_gain = np.ones(len(f))

    def gain(self, P):
        gamma = P / self.noise                                   # a-posteriori SNR
        xi = self.beta * self.prev_s2 / self.noise + (1 - self.beta) * np.maximum(gamma - 1.0, 0.0)
        G = xi / (xi + self.alpha)                               # Wiener with over-subtraction
        G = np.maximum(G, self.floor)
        G[~self.band] = 1.0
        self.prev_s2 = (G ** 2) * P
        self.last_gain = G
        return G

    def process_hop(self, x_hop):
        H, N = self.H, self.N
        self.in_buf[:-H] = self.in_buf[H:]
        self.in_buf[-H:] = x_hop
        X = np.fft.rfft(self.in_buf * self.win, axis=0)          # (bins, C)
        P = np.mean(np.abs(X) ** 2, axis=1)                      # one PSD for all mics
        G = self.gain(P)
        y = np.fft.irfft(X * G[:, None], n=N, axis=0) * self.win  # same gain on every mic
        self.out_buf += y
        out = self.out_buf[:H].copy()
        self.out_buf[:-H] = self.out_buf[H:]
        self.out_buf[-H:] = 0.0
        return out


def filter_signal(x, sr, noise_psd, p):
    L, C = x.shape
    N, H = p["frame"], p["hop"]
    delay = N - H
    total = int(np.ceil((L + delay) / H)) * H
    xp = np.zeros((total, C))
    xp[:L] = x
    wf = WienerPrefilter(noise_psd, sr, C, **p)
    y = np.zeros_like(xp)
    gains = []
    for i in range(0, total, H):
        y[i:i + H] = wf.process_hop(xp[i:i + H])
        gains.append(wf.last_gain.copy())
    return y[delay:delay + L], np.array(gains)


def write_raw_s32(path, y):
    yi = (np.clip(y, -1.0, 1.0 - 2.0 ** -31) * 2.0 ** 31).astype("<i4")
    yi.tofile(path)  # interleaved, little-endian 32-bit



EPS = 1e-12
C_SOUND = 343.0
REPORT = []          # (kind, payload) entries for the HTML report
METRICS = {}


# ================================================================ helpers
def log(step, msg, status=None):
    tag = f"[{status}]" if status else "      "
    line = f"{tag:7s} Step {step}: {msg}"
    print(line)
    REPORT.append(("line", (status, f"Step {step}: {msg}")))


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    REPORT.append(("h2", title))


def save_fig(fig, outdir, name, caption):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    REPORT.append(("img", (path, caption)))
    print(f"        figure -> {path}")


def table(headers, rows):
    REPORT.append(("table", (headers, rows)))
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def read_wav(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x, sr


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_power(x, N, H, chunk=256):
    """Mic-averaged power and mic-0 power, shape (frames, bins)."""
    win = hann(N)[:, None]
    starts = np.arange(0, len(x) - N + 1, H)
    Pavg = np.empty((len(starts), N // 2 + 1))
    P0 = np.empty_like(Pavg)
    for i in range(0, len(starts), chunk):
        idx = starts[i:i + chunk]
        fr = np.stack([x[s:s + N] for s in idx]) * win[None]      # (n, N, C)
        X = np.fft.rfft(fr, axis=1)                               # (n, F, C)
        p = np.abs(X) ** 2
        Pavg[i:i + len(idx)] = p.mean(axis=2)
        P0[i:i + len(idx)] = p[:, :, 0]
    t = (starts + N / 2) / 1.0
    return Pavg, P0, t


def db(x):
    return 10 * np.log10(np.asarray(x) + EPS)


def band_mask(f, p):
    return (f >= p["fmin"]) & (f <= p["fmax"])


def band_energy_per_ch(x, sr, p):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = band_mask(f, p)
    return db(np.sum(np.abs(X[m]) ** 2, axis=0) / len(x))


def sliding_median(v, k=15):
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(vp, k), axis=1)


# ================================================================ geometry / DOA
def parse_cfg_mics(path):
    txt = open(path).read()
    m = txt.find("mics")
    txt = txt[m:] if m >= 0 else txt
    num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    pts = re.findall(r"mu\s*=\s*\(\s*" + num + r"\s*,\s*" + num + r"\s*,\s*" + num + r"\s*\)", txt)
    return np.array([[float(a), float(b), float(c)] for a, b, c in pts])


def direction_grid(pos, n=900):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    g = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    # planar array: front/back ambiguity -> keep one hemisphere around the array normal
    _, _, vt = np.linalg.svd(pos - pos.mean(0))
    normal = vt[-1]
    return g[g @ normal >= 0], normal


def srp_doa(x, sr, pos, p, max_frames=300):
    """SRP-PHAT DOA per analysed frame. Returns (times_s, unit vectors)."""
    N, H = p["frame"], p["hop"]
    f = np.fft.rfftfreq(N, 1 / sr)
    m = band_mask(f, p)
    fb = f[m]
    grid, _ = direction_grid(pos)
    delays = (pos - pos[0]) @ grid.T / C_SOUND                       # (M, G)
    steer = np.exp(-2j * np.pi * fb[None, None, :] * delays.T[:, :, None]).astype(np.complex64)  # (G,M,F)
    starts = np.arange(0, len(x) - N + 1, H)
    # keep frames with energy (top 70 %), then subsample
    win = hann(N)[:, None]
    e = np.array([np.sum(x[s:s + N] ** 2) for s in starts])
    keep = starts[e >= np.percentile(e, 30)]
    keep = keep[:: max(1, len(keep) // max_frames)]
    out, ts = [], []
    for s in keep:
        X = np.fft.rfft(x[s:s + N] * win, axis=0)[m].T               # (M, F)
        Xn = (X / (np.abs(X) + EPS)).astype(np.complex64)
        beam = np.einsum("gmf,mf->gf", np.conj(steer), Xn)
        P = np.sum(np.abs(beam) ** 2, axis=1)
        out.append(grid[np.argmax(P)])
        ts.append((s + N / 2) / sr)
    return np.array(ts), np.array(out)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def mean_dir(v):
    m = v.mean(0)
    return m / (np.linalg.norm(m) + EPS)


def gcc_phat_lag(a, b, sr, p, up=16):
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    R = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    R /= np.abs(R) + EPS
    f = np.fft.rfftfreq(n, 1 / sr)
    R[~band_mask(f, p)] = 0
    cc = np.fft.irfft(R, n * up)
    ms = int(0.002 * sr * up)
    cc = np.concatenate((cc[-ms:], cc[:ms + 1]))
    return (np.argmax(np.abs(cc)) - ms) / (up * sr)


# ================================================================ steps
def step0_inputs(files, p):
    section("STEP 0 - Input check")
    rows, sr_ref = [], None
    for role, path in files:
        x, sr = read_wav(path)
        sr_ref = sr_ref or sr
        clip = np.mean(np.abs(x) > 0.999) * 100
        rms = db(np.mean(x ** 2, axis=0))
        dead = np.where(rms < np.median(rms) - 20)[0]
        rows.append([role, name_of(path), sr, x.shape[1], f"{len(x)/sr:.1f}", f"{clip:.3f}",
                     f"{rms.min():.1f}..{rms.max():.1f}", ",".join(map(str, dead)) or "-"])
        if sr != sr_ref:
            log(0, f"{name_of(path)}: sample rate {sr} differs from {sr_ref}", "FAIL")
        if x.shape[1] != 16:
            log(0, f"{name_of(path)}: {x.shape[1]} channels (expected 16)", "WARN")
        if clip > 0.01:
            log(0, f"{name_of(path)}: {clip:.3f}% clipped samples - lower the gain", "WARN")
        if len(dead):
            log(0, f"{name_of(path)}: mics {list(dead)} are >20 dB below median (dead/blocked?)", "WARN")
    table(["role", "file", "sr", "ch", "dur(s)", "clip%", "mic RMS dBFS", "dead mics"], rows)
    log(0, "inputs read", "PASS")
    return sr_ref


def step1_profile(calib, p, outdir, pos):
    section("STEP 1 - Known-source profile (calibration)")
    x, sr = read_wav(calib)
    N, H = p["frame"], p["hop"]
    Pavg, P0, _ = stft_power(x, N, H)
    f = np.fft.rfftfreq(N, 1 / sr)
    t = (np.arange(len(Pavg)) * H + N / 2) / sr
    m = band_mask(f, p)
    psd = Pavg.mean(0)
    psd_db = db(psd)

    # tonal lines: peaks > 10 dB above a sliding-median floor
    floor_db = sliding_median(psd_db, 15)
    prom = psd_db - floor_db
    is_pk = np.zeros_like(m)
    is_pk[1:-1] = (psd_db[1:-1] > psd_db[:-2]) & (psd_db[1:-1] >= psd_db[2:])
    peaks = np.where(is_pk & m & (prom > 10))[0]
    peaks = peaks[np.argsort(prom[peaks])[::-1]][:10]

    flat = np.exp(np.mean(np.log(psd[m] + EPS))) / (np.mean(psd[m]) + EPS)
    tonal_share = np.sum(psd[peaks]) / np.sum(psd[m]) * 100 if len(peaks) else 0.0
    e_frames = db(Pavg[:, m].sum(1))
    e_std = float(np.std(e_frames))
    # per-bin stability on ~0.2 s averages (removes the random fluctuation of single frames)
    k = max(1, int(0.2 * sr / H))
    Psm = np.lib.stride_tricks.sliding_window_view(Pavg[:, m], k, axis=0).mean(-1)[::k]
    bin_std = np.std(db(Psm), axis=0)
    strong = psd[m] >= np.percentile(psd[m], 75)
    stat_std = float(np.median(bin_std[strong]))
    ch_db = band_energy_per_ch(x, sr, p)
    centroid = np.sum(f[m] * psd[m]) / np.sum(psd[m])

    nature = "TONAL (lines)" if flat < 0.1 and tonal_share > 30 else \
             "MIXED (lines + broadband)" if len(peaks) else "BROADBAND"
    advice = {"TONAL (lines)": "notch is enough; Wiener also works",
              "MIXED (lines + broadband)": "Wiener (or notch first, then Wiener)",
              "BROADBAND": "Wiener is the right filter"}[nature]

    METRICS["profile"] = dict(spectral_flatness=float(flat), tonal_share_pct=float(tonal_share),
                              peaks_hz=[float(f[k]) for k in sorted(peaks)],
                              energy_std_db=e_std, per_bin_std_db=stat_std,
                              centroid_hz=float(centroid), nature=nature,
                              mean_level_dbfs=float(np.mean(e_frames)))
    table(["characteristic", "value"], [
        ["mean in-band level (dB, mic avg)", f"{np.mean(e_frames):.1f}"],
        ["energy stability (std over time)", f"{e_std:.2f} dB"],
        ["per-bin stability (median std)", f"{stat_std:.2f} dB"],
        ["spectral flatness (0 tonal, 1 white)", f"{flat:.3f}"],
        ["share of energy in tonal lines", f"{tonal_share:.1f} %"],
        ["spectral centroid", f"{centroid:.0f} Hz"],
        ["tonal lines (Hz)", ", ".join(f"{f[k]:.0f}" for k in sorted(peaks)) or "none"],
        ["mic energy spread", f"{ch_db.max()-ch_db.min():.1f} dB"],
        ["nature", nature],
        ["recommended", advice]])
    log(1, f"energy std {e_std:.2f} dB", "PASS" if e_std < 1.5 else "WARN" if e_std < 3 else "FAIL")
    if e_std >= 1.5:
        log(1, "known source energy is not constant - the profile will be less accurate", "WARN")
    log(1, f"per-bin std {stat_std:.2f} dB (stationarity of the spectrum)",
        "PASS" if stat_std < 3 else "WARN")

    # figure 1: PSD + lines
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(f, psd_db, lw=1, label="known PSD (mic avg)")
    ax.plot(f, floor_db, lw=0.8, ls="--", label="local floor")
    ax.plot(f[peaks], psd_db[peaks], "rv", label="tonal lines")
    for k in peaks:
        ax.annotate(f"{f[k]:.0f}", (f[k], psd_db[k]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7)
    ax.axvspan(p["fmin"], p["fmax"], color="g", alpha=0.07, label="processing band")
    ax.set_xlim(0, min(8000, sr / 2)); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title("Step 1a - Known-source power spectrum (profile used by the Wiener filter)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save_fig(fig, outdir, "1a_profile_psd.png", "Known-source PSD with tonal lines and processing band")

    # figure 2: spectrogram + energy over time + per-bin variability + per-mic energy
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    S = db(Pavg.T)
    ax[0, 0].imshow(S, origin="lower", aspect="auto", cmap="magma",
                    extent=[t[0], t[-1], f[0], f[-1]], vmin=S.max() - 70, vmax=S.max())
    ax[0, 0].set_ylim(0, 6000); ax[0, 0].set_title("Calibration spectrogram (mic avg)")
    ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("Hz")
    ax[0, 1].plot(t, e_frames, lw=.8)
    ax[0, 1].axhline(np.mean(e_frames), color="r", ls="--", lw=.8)
    ax[0, 1].fill_between(t, np.mean(e_frames) - e_std, np.mean(e_frames) + e_std, alpha=.15, color="r")
    ax[0, 1].set_title(f"In-band energy over time (std {e_std:.2f} dB)")
    ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("dB")
    ax[1, 0].plot(f[m], bin_std, lw=.8)
    ax[1, 0].set_title("Variability per frequency (std of 0.2 s averages, dB)")
    ax[1, 0].set_xlabel("Hz"); ax[1, 0].set_ylabel("dB")
    if pos is not None and len(pos) == len(ch_db):
        ax_ids = np.argsort(np.ptp(pos, axis=0))[::-1][:2]      # the two in-plane cfg axes
        uv = pos[:, np.sort(ax_ids)]
        sc = ax[1, 1].scatter(uv[:, 0] * 1000, uv[:, 1] * 1000, c=ch_db, s=400, cmap="viridis",
                              vmin=min(ch_db.min(), ch_db.mean() - 1), vmax=max(ch_db.max(), ch_db.mean() + 1))
        for i, (a, b) in enumerate(uv * 1000):
            ax[1, 1].text(a, b, str(i), ha="center", va="center", color="w", fontsize=8)
        cb = fig.colorbar(sc, ax=ax[1, 1], label="dB")
        cb.formatter.set_useOffset(False); cb.update_ticks()
        lab = "xyz"
        ax[1, 1].set_aspect("equal")
        ax[1, 1].set_xlabel(f"cfg {lab[min(ax_ids)]} (mm)"); ax[1, 1].set_ylabel(f"cfg {lab[max(ax_ids)]} (mm)")
    else:
        ax[1, 1].bar(range(len(ch_db)), ch_db)
        ax[1, 1].set_xlabel("mic")
    ax[1, 1].set_title("Known-source energy per mic (in band)")
    save_fig(fig, outdir, "1b_profile_characteristics.png",
             "Calibration spectrogram, energy stability, per-frequency variability, per-mic energy")

    # profile for the filter: same sqrt-Hann window as prefilter.py
    w = sqrt_hann(N)[:, None]
    prof = np.mean([np.mean(np.abs(np.fft.rfft(x[s:s + N] * w, axis=0)) ** 2, axis=1)
                    for s in range(0, len(x) - N + 1, H)], axis=0)
    np.savez(os.path.join(outdir, "known_profile.npz"), noise_psd=prof, sr=sr, frame=N, hop=H)
    log(1, "profile saved -> known_profile.npz", "PASS")
    return psd, f


def step2_overlap(target, known_psd, f, p, outdir):
    section("STEP 2 - Spectral overlap known vs target")
    x, sr = read_wav(target)
    Pavg, _, _ = stft_power(x, p["frame"], p["hop"])
    tpsd = Pavg.mean(0)
    m = band_mask(f, p)
    dominated = (known_psd > tpsd) & m
    share = np.sum(tpsd[dominated]) / np.sum(tpsd[m]) * 100
    METRICS["overlap_target_energy_in_known_bins_pct"] = float(share)
    table(["quantity", "value"], [
        ["target energy in bins where known > target", f"{share:.1f} %"],
        ["bins dominated by known source", f"{dominated.sum()} / {m.sum()}"]])
    log(2, f"{share:.1f}% of target energy lies in known-dominated bins "
           f"(this part will be attenuated)", "PASS" if share < 20 else "WARN")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(f, db(known_psd), label="known (S0)")
    ax.plot(f, db(tpsd), label="target (S2)")
    ax.fill_between(f, -200, 200, where=dominated, color="r", alpha=.1, label="known dominates")
    ax.set_ylim(min(db(tpsd[m]).min(), db(known_psd[m]).min()) - 5,
                max(db(tpsd).max(), db(known_psd).max()) + 5)
    ax.set_xlim(0, min(8000, sr / 2)); ax.axvspan(p["fmin"], p["fmax"], color="g", alpha=.05)
    ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Step 2 - Known vs target spectra (red: bins the filter will cut)")
    save_fig(fig, outdir, "2_overlap.png", "Spectral overlap between known source and target")


def step3_filter(files, p, outdir):
    section("STEP 3 - Wiener filtering (common gain per bin, all 16 mics)")
    prof = np.load(os.path.join(outdir, "known_profile.npz"))
    out = {}
    rows = []
    for role, path in files:
        x, sr = read_wav(path)
        t0 = time.perf_counter()
        y, gains = filter_signal(x, sr, prof["noise_psd"], p)
        dt = time.perf_counter() - t0
        hops = int(np.ceil(len(x) / p["hop"]))
        ms = dt / hops * 1000
        budget = p["hop"] / sr * 1000
        stem = name_of(path) + "_filtA"
        sf.write(os.path.join(outdir, stem + ".wav"), y, sr, subtype="PCM_32")
        write_raw_s32(os.path.join(outdir, stem + ".raw"), y)
        out[path] = (x, y, gains, sr)
        rows.append([role, stem, f"{ms:.2f}", f"{budget:.1f}", "OK" if ms < budget * 0.5 else "TIGHT"])
    table(["role", "output", "ms/hop", "budget ms", "real time"], rows)
    METRICS["cpu_ms_per_hop_laptop"] = float(rows[-1][2])
    log(3, "filtered WAV + RAW (S32_LE) written for every file", "PASS")
    return out


def step4_gains(results, files, p, outdir, known_psd, f):
    section("STEP 4 - Gain behaviour")
    fig, ax = plt.subplots(len(files), 1, figsize=(11, 2.6 * len(files)), squeeze=False)
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    for i, (role, path) in enumerate(files):
        x, y, g, sr = results[path]
        t = np.arange(len(g)) * p["hop"] / sr
        im = ax[i, 0].imshow(20 * np.log10(g.T + EPS), origin="lower", aspect="auto", cmap="viridis",
                             extent=[t[0], t[-1], f[0], f[-1]], vmin=-32, vmax=0)
        ax[i, 0].set_ylim(0, min(p["fmax"] * 1.3, sr / 2))
        ax[i, 0].set_title(f"{role}: {name_of(path)} - gain (dB)")
        ax[i, 0].set_ylabel("Hz")
        fig.colorbar(im, ax=ax[i, 0], label="dB")
        ax2.plot(f, 20 * np.log10(g.mean(0) + EPS), label=f"{role}: {name_of(path)}")
    ax[-1, 0].set_xlabel("s")
    save_fig(fig, outdir, "4a_gain_maps.png", "Wiener gain over time and frequency for every file")
    ax3 = ax2.twinx()
    ax3.plot(f, db(known_psd), color="k", lw=.6, alpha=.4, label="known PSD")
    ax2.set_xlim(0, min(p["fmax"] * 1.3, f[-1])); ax2.set_ylim(-35, 2)
    ax2.set_xlabel("Hz"); ax2.set_ylabel("mean gain (dB)"); ax2.grid(alpha=.3)
    ax2.legend(fontsize=8, loc="lower right"); ax3.set_ylabel("known PSD (dB)")
    ax2.set_title("Step 4b - Mean gain vs frequency (dips should match the known spectrum)")
    save_fig(fig2, outdir, "4b_mean_gain.png", "Mean gain per frequency, with the known PSD overlaid")
    log(4, "gain maps written", "PASS")


def step5_spectrograms(results, files, p, outdir):
    section("STEP 5 - Spectrograms before / after")
    for role, path in files:
        x, y, _, sr = results[path]
        Px, _, _ = stft_power(x, p["frame"], p["hop"])
        Py, _, _ = stft_power(y, p["frame"], p["hop"])
        f = np.fft.rfftfreq(p["frame"], 1 / sr)
        t = np.arange(len(Px)) * p["hop"] / sr
        Sx, Sy = db(Px.T), db(Py.T)
        vmax = Sx.max()
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for a, S, ttl in ((ax[0], Sx, "BEFORE"), (ax[1], Sy, "AFTER Wiener")):
            im = a.imshow(S, origin="lower", aspect="auto", cmap="magma",
                          extent=[t[0], t[-1], f[0], f[-1]], vmin=vmax - 70, vmax=vmax)
            a.set_title(f"{role}: {name_of(path)} - {ttl} (mic avg)")
            a.set_ylabel("Hz"); a.set_ylim(0, 6000)
            a.axhline(p["fmin"], color="c", lw=.5, ls="--"); a.axhline(p["fmax"], color="c", lw=.5, ls="--")
            plt.colorbar(im, ax=a, label="dB")
        from matplotlib.colors import TwoSlopeNorm
        im = ax[2].imshow(np.clip(Sy - Sx, -40, 5), origin="lower", aspect="auto", cmap="RdBu_r",
                          extent=[t[0], t[-1], f[0], f[-1]],
                          norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=5))
        ax[2].set_title("DIFFERENCE after - before (dB): blue = removed")
        ax[2].set_ylabel("Hz"); ax[2].set_xlabel("s"); ax[2].set_ylim(0, 6000)
        plt.colorbar(im, ax=ax[2], label="dB")
        save_fig(fig, outdir, f"5_spec_{name_of(path)}.png", f"Spectrogram before/after: {name_of(path)}")
    log(5, "spectrograms written", "PASS")


def step6_energy(results, files, p, outdir):
    section("STEP 6 - In-band energy over time")
    fig, ax = plt.subplots(len(files), 1, figsize=(11, 2.4 * len(files)), squeeze=False, sharex=True)
    for i, (role, path) in enumerate(files):
        x, y, _, sr = results[path]
        f = np.fft.rfftfreq(p["frame"], 1 / sr)
        m = band_mask(f, p)
        Px, _, _ = stft_power(x, p["frame"], p["hop"])
        Py, _, _ = stft_power(y, p["frame"], p["hop"])
        t = np.arange(len(Px)) * p["hop"] / sr
        ex, ey = db(Px[:, m].sum(1)), db(Py[:, m].sum(1))
        ax[i, 0].plot(t, ex, lw=.8, label="before")
        ax[i, 0].plot(t, ey, lw=.8, label="after")
        ax[i, 0].set_title(f"{role}: {name_of(path)}  (mean drop {np.mean(ex-ey):.1f} dB)")
        ax[i, 0].set_ylabel("dB"); ax[i, 0].grid(alpha=.3); ax[i, 0].legend(fontsize=7)
    ax[-1, 0].set_xlabel("s")
    save_fig(fig, outdir, "6_energy_time.png", "In-band energy before vs after for every file")
    log(6, "energy curves written", "PASS")


def step7_metrics(results, files, p):
    section("STEP 7 - Metrics")
    rows = []
    for role, path in files:
        x, y, _, sr = results[path]
        red = band_energy_per_ch(x, sr, p) - band_energy_per_ch(y, sr, p)
        rows.append([role, name_of(path), f"{red.mean():.1f}", f"{red.min():.1f}", f"{red.max():.1f}"])
        METRICS.setdefault("reduction_db", {})[name_of(path)] = float(red.mean())
        if role == "known":
            st = "PASS" if red.mean() >= 20 else "WARN" if red.mean() >= 12 else "FAIL"
            log(7, f"known source reduced by {red.mean():.1f} dB (goal >= 20 dB)", st)
            if st != "PASS":
                log(7, "-> raise --alpha (2-3) or lower --floor (0.01)", None)
        if role == "target":
            st = "PASS" if red.mean() <= 3 else "WARN" if red.mean() <= 6 else "FAIL"
            log(7, f"target lost {red.mean():.1f} dB (goal <= 3 dB)", st)
            if st != "PASS":
                log(7, "-> lower --alpha (~1.0) or raise --floor (0.05-0.1)", None)
        if role == "calib":
            log(7, f"calibration reduced by {red.mean():.1f} dB (self-test)", None)
    table(["role", "file", "mean drop dB", "min (mic)", "max (mic)"], rows)


def step8_direction(results, files, p, pos, outdir):
    section("STEP 8 - Direction check (phase preserved?)")
    tol = 0.042 * np.sin(np.radians(3)) / C_SOUND
    rows = []
    for role, path in files:
        x, y, _, sr = results[path]
        L = min(len(x), int(10 * sr))
        worst = max(abs(gcc_phat_lag(y[:L, 0], y[:L, m], sr, p) - gcc_phat_lag(x[:L, 0], x[:L, m], sr, p))
                    for m in range(1, x.shape[1]))
        rows.append([role, name_of(path), f"{worst*1e6:.1f}", "PASS" if worst <= tol else "CHANGED"])
    table(["role", "file", "max GCC delay change (us)", "verdict"], rows)
    log(8, f"GCC-PHAT tolerance {tol*1e6:.1f} us (= 3 deg on 42 mm). "
           "On target-only (S2) it must PASS; on mixes a change is expected "
           "(the dominant source switches)", None)

    if pos is None or len(pos) != x.shape[1]:
        log(8, "no --cfg (or mic count mismatch): SRP-PHAT DOA skipped", "WARN")
        return
    doa = {}
    for role, path in files:
        x, y, _, sr = results[path]
        tb, db_ = srp_doa(x, sr, pos, p)
        ta, da = srp_doa(y, sr, pos, p)
        doa[path] = (role, tb, db_, ta, da)
    known_dir = next((mean_dir(v[2]) for v in doa.values() if v[0] == "known"), None)
    target_dir = next((mean_dir(v[2]) for v in doa.values() if v[0] == "target"), None)
    if known_dir is not None and target_dir is not None:
        sep = angle_deg(known_dir, target_dir)
        log(8, f"reference directions: known vs target separated by {sep:.1f} deg", None)

    rows = []
    fig, ax = plt.subplots(len(doa), 1, figsize=(11, 2.5 * len(doa)), squeeze=False, sharex=True)
    for i, (path, (role, tb, dbv, ta, dav)) in enumerate(doa.items()):
        n = min(len(dbv), len(dav))
        change = angle_deg(dbv[:n], dav[:n])
        r = [role, name_of(path), f"{np.median(change):.1f}"]
        if known_dir is not None:
            kb = np.mean(angle_deg(dbv, known_dir) < 20) * 100
            ka = np.mean(angle_deg(dav, known_dir) < 20) * 100
            r += [f"{kb:.0f}", f"{ka:.0f}"]
            ax[i, 0].plot(tb, angle_deg(dbv, known_dir), ".", ms=3, label="before: angle to known")
            ax[i, 0].plot(ta, angle_deg(dav, known_dir), ".", ms=3, label="after: angle to known")
        if target_dir is not None and role != "known":
            tb_ = np.mean(angle_deg(dbv, target_dir) < 20) * 100
            ta_ = np.mean(angle_deg(dav, target_dir) < 20) * 100
            r += [f"{tb_:.0f}", f"{ta_:.0f}"]
        else:
            r += ["-", "-"]
        rows.append(r)
        if role == "target":
            st = "PASS" if np.median(change) < 3 else "FAIL"
            log(8, f"target-only DOA change median {np.median(change):.1f} deg (goal < 3)", st)
        ax[i, 0].axhline(20, color="r", ls="--", lw=.6)
        ax[i, 0].set_title(f"{role}: {name_of(path)} - SRP-PHAT DOA, angle to known direction")
        ax[i, 0].set_ylabel("deg"); ax[i, 0].legend(fontsize=7); ax[i, 0].grid(alpha=.3)
    ax[-1, 0].set_xlabel("s")
    save_fig(fig, outdir, "8_doa.png",
             "Dominant direction before/after; below the red line = pointing at the known source")
    table(["role", "file", "median DOA change deg", "% on known before", "% on known after",
           "% on target before", "% on target after"], rows)
    METRICS["doa"] = rows
    log(8, "mixes: '% on known' should DROP and '% on target' should RISE after filtering", None)


# ================================================================ HTML report
def write_report(outdir, p):
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;"
           "background:#fafafa;color:#222}h1{font-size:1.5em}h2{border-bottom:2px solid #ccc;"
           "padding-top:18px}img{max-width:100%;border:1px solid #ddd;background:#fff}"
           "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 8px;"
           "font-size:.9em}th{background:#eee}.PASS{color:#080;font-weight:bold}"
           ".WARN{color:#b60;font-weight:bold}.FAIL{color:#c00;font-weight:bold}"
           "p.cap{font-size:.85em;color:#555;margin-top:2px}")
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>Wiener before SSL - analysis</title>"
         f"<style>{css}</style></head><body><h1>Wiener filter before SSL - step-by-step analysis</h1>"
         f"<p>Parameters: {p}</p>"]
    for kind, val in REPORT:
        if kind == "h2":
            h.append(f"<h2>{val}</h2>")
        elif kind == "line":
            st, msg = val
            h.append(f"<div>{'<span class=' + st + '>[' + st + ']</span> ' if st else ''}{msg}</div>")
        elif kind == "table":
            hd, rows = val
            h.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in hd) + "</tr>" +
                     "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) +
                     "</table>")
        elif kind == "img":
            path, cap = val
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            h.append(f"<img src='data:image/png;base64,{b64}'><p class='cap'>{cap}</p>")
    h.append("</body></html>")
    out = os.path.join(outdir, "report.html")
    open(out, "w").write("\n".join(h))
    json.dump(METRICS, open(os.path.join(outdir, "metrics.json"), "w"), indent=2, default=str)
    print(f"\nReport -> {out}\nMetrics -> {os.path.join(outdir, 'metrics.json')}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Step-by-step analysis of the Wiener filter before SSL")
    ap.add_argument("--calib", required=True, help="S0: known source only (profile)")
    ap.add_argument("--known", help="S1: known source only (test)")
    ap.add_argument("--target", help="S2: target only")
    ap.add_argument("--mix", nargs="*", default=[], help="S3/S4/S5: both sources")
    ap.add_argument("--cfg", help="ODAS cfg (mic positions) for SRP-PHAT DOA check")
    ap.add_argument("--outdir", default="analysis_A")
    for k, v in DEFAULTS.items():
        if k == "full_band":
            ap.add_argument("--full-band", dest="full_band", action="store_true")
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in DEFAULTS}
    os.makedirs(a.outdir, exist_ok=True)

    files = [("calib", a.calib)]
    if a.known: files.append(("known", a.known))
    if a.target: files.append(("target", a.target))
    files += [("mix", m) for m in a.mix]

    pos = None
    if a.cfg:
        pos = parse_cfg_mics(a.cfg)
        print(f"cfg: {len(pos)} mic positions read from {a.cfg}")

    step0_inputs(files, p)
    known_psd, f = step1_profile(a.calib, p, a.outdir, pos)
    if a.target:
        step2_overlap(a.target, known_psd, f, p, a.outdir)
    results = step3_filter(files, p, a.outdir)
    step4_gains(results, files, p, a.outdir, known_psd, f)
    step5_spectrograms(results, files, p, a.outdir)
    step6_energy(results, files, p, a.outdir)
    step7_metrics(results, files, p)
    step8_direction(results, files, p, pos, a.outdir)
    write_report(a.outdir, p)


if __name__ == "__main__":
    main()
