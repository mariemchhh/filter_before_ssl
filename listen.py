#!/usr/bin/env python3
"""
listen.py - Before/after listening for the Wiener test.

  python3 listen.py --dataset ~/dataset --analysis ~/dataset/analysis_A
  xdg-open ~/dataset/analysis_A/listen/listen.html

For every S*.wav that has an S*_filtA.wav, writes (mono, 16-bit, same gain for before and after):
  S*_before.wav   S*_after.wav   S*_AB.wav (3 s before / 3 s after, alternating)
and listen.html with players + the before/after spectrograms.
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf


def mono(x, mic):
    return x[:, mic] if mic >= 0 else x.mean(axis=1)   # -1 = average of the 16 mics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.expanduser("~/dataset"))
    ap.add_argument("--analysis", default=os.path.expanduser("~/dataset/analysis_A"))
    ap.add_argument("--mic", type=int, default=0, help="mic to listen to (-1 = average of all mics)")
    ap.add_argument("--ab", type=float, default=3.0, help="segment length (s) for the A/B file")
    a = ap.parse_args()

    out = os.path.join(a.analysis, "listen")
    os.makedirs(out, exist_ok=True)
    rows = []
    for fa in sorted(glob.glob(os.path.join(a.analysis, "*_filtA.wav"))):
        stem = os.path.basename(fa)[:-len("_filtA.wav")]
        fb = os.path.join(a.dataset, stem + ".wav")
        if not os.path.exists(fb):
            print(f"[skip] {stem}: original {fb} not found")
            continue
        xb, sr = sf.read(fb, dtype="float64", always_2d=True)
        xa, _ = sf.read(fa, dtype="float64", always_2d=True)
        b, f = mono(xb, a.mic), mono(xa, a.mic)
        n = min(len(b), len(f))
        b, f = b[:n] - b[:n].mean(), f[:n] - f[:n].mean()
        g = 0.9 / (np.max(np.abs(b)) + 1e-12)            # SAME gain: the level drop stays audible
        b, f = b * g, np.clip(f * g, -1, 1)

        seg, gap = int(a.ab * sr), np.zeros(int(0.25 * sr))
        ab = []
        for i in range(0, n - seg + 1, 2 * seg):
            ab += [b[i:i + seg], gap, f[i + seg:i + 2 * seg] if i + 2 * seg <= n else f[i:i + seg], gap]
        ab = np.concatenate(ab) if ab else b

        for tag, sig in (("before", b), ("after", f), ("AB", ab)):
            sf.write(os.path.join(out, f"{stem}_{tag}.wav"), sig, sr, subtype="PCM_16")
        rb = 20 * np.log10(np.sqrt(np.mean(b ** 2)) + 1e-12)
        rf = 20 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-12)
        rows.append((stem, rb, rf))
        print(f"{stem:12s} before {rb:6.1f} dBFS   after {rf:6.1f} dBFS   drop {rb - rf:5.1f} dB")

    what = {"S0_calib": "known only (calibration) - should almost disappear",
            "S1_known": "known only - should almost disappear",
            "S2_target": "target only - should sound almost the SAME",
            "S3_both": "both - known removed, target kept",
            "S4_cross": "target crossing the known direction - target must stay audible"}
    mic = "average of 16 mics" if a.mic < 0 else f"mic {a.mic}"
    h = ["<!doctype html><html><head><meta charset='utf-8'><title>Wiener - listen</title><style>"
         "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}"
         "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
         "th{background:#eee}audio{width:100%}img{width:100%;margin-top:6px}</style></head><body>",
         f"<h1>Wiener before SSL - before / after</h1><p>Listening to {mic}. Before and after use the "
         "same gain, so the level drop you hear is real. A/B file: 3 s before, short silence, 3 s after, "
         "repeated. Use headphones.</p>",
         "<table><tr><th>Scenario</th><th>Before</th><th>After</th><th>A/B</th></tr>"]
    for stem, rb, rf in rows:
        h.append(f"<tr><td><b>{stem}</b><br>{what.get(stem, '')}<br>drop: {rb - rf:.1f} dB</td>"
                 f"<td><audio controls src='{stem}_before.wav'></audio></td>"
                 f"<td><audio controls src='{stem}_after.wav'></audio></td>"
                 f"<td><audio controls src='{stem}_AB.wav'></audio></td></tr>"
                 f"<tr><td colspan=4><img src='../5_spec_{stem}.png'></td></tr>")
    h.append("</table></body></html>")
    page = os.path.join(out, "listen.html")
    open(page, "w").write("\n".join(h))
    print(f"\nOpen: xdg-open {page}")


if __name__ == "__main__":
    main()
