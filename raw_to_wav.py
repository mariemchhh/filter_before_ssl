#!/usr/bin/env python3
"""raw_to_wav.py <dir> [--normalize]
Converts raw SSS captures to .wav. Supports:
  - sss_raw_record.py output : sss_src<N>.raw or sss_all.raw + sss_meta.txt
  - record_sss.py output     : <stream>_src<N>.raw + meta.txt
"""
import argparse, glob, os, sys, wave
import numpy as np

DT = {8: np.int8, 16: np.int16, 32: np.int32}

def read_meta(path):
    m = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                m[k] = v
    return m

def write_wav(path, data, fs, width):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(width); w.setframerate(fs)
        w.writeframes(data.tobytes())

def convert(x, fs, bits, base, normalize):
    out = base + ".wav"
    write_wav(out, x, fs, bits // 8)
    peak = np.max(np.abs(x.astype(np.float64))) / 2 ** (bits - 1) if len(x) else 0.0
    msg = f"{os.path.basename(out):28s} {len(x)/fs:6.1f} s  peak {20*np.log10(peak+1e-12):6.1f} dBFS"
    if normalize and peak > 0:
        y = x.astype(np.float64) / 2 ** (bits - 1) * (10 ** (-3 / 20) / peak)
        write_wav(base + "_norm.wav", (np.clip(y, -1, 1) * 32767).astype(np.int16), fs, 2)
        msg += "  (+ _norm.wav)"
    print(msg)

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--normalize", action="store_true",
                help="also write *_norm.wav scaled to -3 dBFS peak (for listening)")
a = ap.parse_args()
d = os.path.expanduser(a.dir)

if os.path.isfile(os.path.join(d, "sss_meta.txt")):          # sss_raw_record.py
    m = read_meta(os.path.join(d, "sss_meta.txt"))
    fs = int(float(m["sampling_frequency"]))
    bits = int(m["dtype"].replace("int", ""))
    ch = int(m["channel_count"])
    if m.get("layout") == "interleaved":
        x = np.fromfile(os.path.join(d, "sss_all.raw"), dtype=DT[bits])
        x = x[:(len(x) // ch) * ch].reshape(-1, ch)
        for c in range(ch):
            convert(np.ascontiguousarray(x[:, c]), fs, bits,
                    os.path.join(d, f"sss_src{c}"), a.normalize)
    else:
        for p in sorted(glob.glob(os.path.join(d, "sss_src*.raw"))):
            convert(np.fromfile(p, dtype=DT[bits]), fs, bits, p[:-4], a.normalize)

elif os.path.isfile(os.path.join(d, "meta.txt")):             # record_sss.py
    m = read_meta(os.path.join(d, "meta.txt"))
    for p in sorted(glob.glob(os.path.join(d, "*_src*.raw"))):
        stream = os.path.basename(p).split("_src")[0]
        fs = int(m[f"{stream}_fs"])
        bits = int(m[f"{stream}_dtype"].replace("int", ""))
        convert(np.fromfile(p, dtype=DT[bits]), fs, bits, p[:-4], a.normalize)
else:
    sys.exit(f"no sss_meta.txt or meta.txt in {d}")
