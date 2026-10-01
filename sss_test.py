#!/usr/bin/env python3
"""
sss_test.py - New test: Wiener filter BEFORE SSL, then ODAS, and save the 4 SSS outputs as WAV.

Standalone (needs only numpy<2, soundfile, matplotlib, arecord, odaslive).

Record new takes (guided):
  cd ~/Downloads
  python3 sss_test.py --cfg ~/Downloads/uma16.cfg --vertical \
      --odas ~/odas_ws/odas/build/bin/odaslive

Reuse existing takes instead of recording (looks for *calib*.wav, *known*.wav, ... in the folder):
  python3 sss_test.py --cfg ~/Downloads/uma16.cfg --vertical \
      --odas ~/odas_ws/odas/build/bin/odaslive --from ~/dataset

Flow per scene (calib, known, target, both, cross):
  arecord 16 mics -> Wiener (one common gain per bin, all 16 mics) -> ODAS on RAW and on FILTERED
Saved in ~/dataset/sss_tests/<date_time>/<scene>/ :
  mics_raw.wav, mics_filt.wav          16-ch input, before / after Wiener
  sss_raw_4ch.wav, sss_filt_4ch.wav    the 4 SSS outputs, true levels (4 channels)
  sss_raw_ch0..3.wav, sss_filt_ch0..3.wav   each SSS output alone, same gain raw/filt (to listen)
  tracks_raw.csv, tracks_filt.csv      SST per frame: slot, id, x, y, z, az, el, activity
  sss_spectrograms.png, tracks.png     figures
  odas_raw.log, odas_filt.log
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
SCENES = [("calib", "KNOWN source ON alone at its fixed position (learns the filter profile)."),
          ("known", "KNOWN source ON alone, same position (test)."),
          ("target", "Known OFF. TARGET ON alone, away from the known source."),
          ("both", "Known ON + TARGET ON, target away from the known direction."),
          ("cross", "Known ON + TARGET moving slowly ACROSS the known direction.")]
SINKS = {"potential": "potential.json", "tracked": "tracked.json",
         "separated": "separated.raw", "postfiltered": "postfiltered.raw"}


# ================================================================ Wiener (same as prefilter.py)
def sqrt_hann(n):
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))


def wiener_profile(x, N, H):
    w = sqrt_hann(N)[:, None]
    return np.mean([np.mean(np.abs(np.fft.rfft(x[s:s + N] * w, axis=0)) ** 2, axis=1)
                    for s in range(0, len(x) - N + 1, H)], axis=0)


def wiener_filter(x, sr, noise, p):
    N, H = p["frame"], p["hop"]
    L, C = x.shape
    win = sqrt_hann(N)[:, None]
    f = np.fft.rfftfreq(N, 1 / sr)
    band = np.ones_like(f, bool) if p["full_band"] else (f >= p["fmin"]) & (f <= p["fmax"])
    noise = np.maximum(noise, EPS)
    delay = N - H
    total = int(np.ceil((L + delay) / H)) * H
    xp = np.zeros((total, C)); xp[:L] = x
    y = np.zeros_like(xp)
    inb, outb, prev = np.zeros((N, C)), np.zeros((N, C)), np.zeros(len(f))
    for i in range(0, total, H):
        inb[:-H] = inb[H:]; inb[-H:] = xp[i:i + H]
        X = np.fft.rfft(inb * win, axis=0)
        P = np.mean(np.abs(X) ** 2, axis=1)
        gamma = P / noise
        xi = p["beta"] * prev / noise + (1 - p["beta"]) * np.maximum(gamma - 1, 0)
        G = np.maximum(xi / (xi + p["alpha"]), p["floor"]); G[~band] = 1.0
        prev = G ** 2 * P
        outb += np.fft.irfft(X * G[:, None], n=N, axis=0) * win
        y[i:i + H] = outb[:H]
        outb[:-H] = outb[H:]; outb[-H:] = 0
    return y[delay:delay + L]


# ================================================================ cfg / ODAS
def _find(txt, key):
    m = re.search(r"\b%s\s*:\s*\{" % key, txt)
    if not m:
        return None
    im = re.compile(r"interface\s*:\s*\{[^{}]*\}\s*;?").search(txt, m.end())
    return (m, im) if im else None


def cfg_info(txt):
    info = {}
    for key in ["raw"] + list(SINKS):
        r = _find(txt, key)
        if r:
            seg = txt[r[0].end():r[1].start()]
            info[key] = {}
            for k in ("fS", "hopSize", "nBits", "nChannels"):
                m = re.search(r"\b%s\s*=\s*([-\d.]+)" % k, seg)
                info[key][k] = float(m.group(1)) if m else None
    for k in ("raw", "tracked", "separated"):
        if k not in info:
            sys.exit(f"[FAIL] cfg: section '{k}' with an interface block not found")
    return info


def make_cfg(txt, raw_in, d):
    for key in ["raw"] + list(SINKS):
        r = _find(txt, key)
        if r:
            path = raw_in if key == "raw" else os.path.join(d, SINKS[key])
            txt = txt[:r[1].start()] + f'interface: {{ type = "file"; path = "{path}"; }};' + txt[r[1].end():]
    return txt


def run_odas(odas, cfg, timeout, log):
    with open(log, "w") as lf:
        p = subprocess.Popen([odas, "-c", cfg], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for s in (signal.SIGINT, signal.SIGKILL):
                try:
                    os.killpg(p.pid, s)
                except ProcessLookupError:
                    break
                time.sleep(2)


def read_tracked(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    dec, i, fr = json.JSONDecoder(), 0, []
    while True:
        i = txt.find("{", i)
        if i < 0:
            break
        try:
            o, i = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            break
        if isinstance(o, dict) and "src" in o:
            fr.append(o)
    if not fr:
        return None
    T, S = len(fr), max(len(f["src"]) for f in fr)
    ids, xyz, act = np.zeros((T, S), int), np.zeros((T, S, 3)), np.zeros((T, S))
    for t, f in enumerate(fr):
        for s, src in enumerate(f["src"]):
            ids[t, s] = int(src.get("id", 0))
            xyz[t, s] = [src.get("x", 0.0), src.get("y", 0.0), src.get("z", 0.0)]
            act[t, s] = float(src.get("activity", 0.0))
    return dict(ids=ids, xyz=xyz, act=act)


def read_sep(path, nbits, nch):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = np.fromfile(path, dtype="<i2" if nbits == 16 else "<i4")
    n = len(d) // nch * nch
    return d[:n].reshape(-1, nch) / float(2 ** (nbits - 1))


def write_pcm(path, x, nbits):
    if nbits == 16:
        (np.clip(x, -1, 1 - 2 ** -15) * 2 ** 15).astype("<i2").tofile(path)
    else:
        (np.clip(x, -1, 1 - 2 ** -31) * 2 ** 31).astype("<i4").tofile(path)


# ================================================================ geometry
def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def az_el(xyz, vertical):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if vertical:
        return np.degrees(np.arctan2(x, z)), np.degrees(np.arctan2(y, np.hypot(x, z)))
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.hypot(x, y)))


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


# ================================================================ recording
def record(path, dur, dev, sr):
    # free the card: match the exact process NAME (-x), never the command line
    # (-f would match this script itself, whose arguments contain "odaslive")
    for k in ("odas_core_node", "odaslive"):
        subprocess.run(["pkill", "-9", "-x", k], stderr=subprocess.DEVNULL)
    base = ["arecord", "-D", dev, "-c", "16", "-r", str(sr), "-f", "S32_LE", "-d", str(int(dur)), path]
    cmd = (["pasuspender", "--"] + base) if shutil.which("pasuspender") else base
    if subprocess.run(cmd).returncode != 0 and cmd[0] == "pasuspender":
        subprocess.run(base)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        sys.exit(f"[FAIL] recording failed: {path}  (check 'arecord -l' and --device)")


def find_take(folder, name):
    c = sorted(glob.glob(os.path.join(folder, f"*{name}*.wav")))
    c = [f for f in c if "filt" not in os.path.basename(f)]
    return c[0] if c else None


# ================================================================ per scene
def process_scene(name, wav, sd, a, p, info, base_cfg, noise, known_ref):
    x, sr = sf.read(wav, dtype="float64", always_2d=True)
    rch, rfs, rb = int(info["raw"]["nChannels"] or 16), int(info["raw"]["fS"] or 44100), int(info["raw"]["nBits"] or 32)
    if x.shape[1] != rch or sr != rfs:
        sys.exit(f"[FAIL] {wav}: {x.shape[1]} ch / {sr} Hz, cfg raw wants {rch} ch / {rfs} Hz")
    sb = int(info["separated"]["nBits"] or 16)
    sfs = int(info["separated"]["fS"] or rfs)
    shop = int(info["separated"]["hopSize"] or info["raw"]["hopSize"] or 512)

    if noise is None:                          # calib scene: learn the profile from itself
        noise = wiener_profile(x, p["frame"], p["hop"])
        np.savez(os.path.join(os.path.dirname(sd), "known_profile.npz"), noise_psd=noise, sr=sr,
                 frame=p["frame"], hop=p["hop"])
    y = wiener_filter(x, sr, noise, p)
    sf.write(os.path.join(sd, "mics_raw.wav"), x, sr, subtype="PCM_32")
    sf.write(os.path.join(sd, "mics_filt.wav"), y, sr, subtype="PCM_32")

    out = {}
    for ver, sig in (("raw", x), ("filt", y)):
        d = os.path.join(sd, "odas_" + ver)
        os.makedirs(d, exist_ok=True)
        for f in SINKS.values():
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
        pcm = os.path.join(d, "input.raw")
        write_pcm(pcm, sig, rb)
        cfgp = os.path.join(d, "odas_file.cfg")
        open(cfgp, "w").write(make_cfg(base_cfg, pcm, d))
        run_odas(a.odas, cfgp, len(sig) / sr + 30, os.path.join(sd, f"odas_{ver}.log"))
        tr = read_tracked(os.path.join(d, SINKS["tracked"]))
        nch = tr["ids"].shape[1] if tr else 4
        sep = read_sep(os.path.join(d, SINKS["separated"]), sb, nch)
        os.remove(pcm)                          # big temporary file
        out[ver] = (tr, sep)
        if tr is None or sep is None:
            print(f"   [NO OUTPUT] ODAS {ver}: see {sd}/odas_{ver}.log")

    # ---- save SSS wavs + tracks
    peak = max((np.max(np.abs(s)) for _, s in out.values() if s is not None and s.size), default=0)
    g = 0.9 / peak if peak > 0 else 1.0
    for ver, (tr, sep) in out.items():
        if sep is not None:
            sf.write(os.path.join(sd, f"sss_{ver}_4ch.wav"), sep, sfs, subtype="PCM_16")
            for c in range(sep.shape[1]):
                sf.write(os.path.join(sd, f"sss_{ver}_ch{c}.wav"), np.clip(sep[:, c] * g, -1, 1), sfs,
                         subtype="PCM_16")
        if tr is not None:
            az, el = az_el(tr["xyz"], a.vertical)
            with open(os.path.join(sd, f"tracks_{ver}.csv"), "w") as f:
                f.write("frame,time_s,slot,id,x,y,z,az_deg,el_deg,activity\n")
                T, S = tr["ids"].shape
                for t in range(T):
                    for s in range(S):
                        if tr["ids"][t, s]:
                            xx, yy, zz = tr["xyz"][t, s]
                            f.write(f"{t},{t*shop/sfs:.3f},{s},{tr['ids'][t,s]},{xx:.3f},{yy:.3f},{zz:.3f},"
                                    f"{az[t,s]:.1f},{el[t,s]:.1f},{tr['act'][t,s]:.3f}\n")

    # ---- known reference = main track of the calib baseline
    if name == "calib" and out["raw"][0] is not None:
        tr = out["raw"][0]
        act = tr["ids"] != 0
        s = int(np.argmax(act.mean(0)))
        if act[:, s].any():
            known_ref = unit(tr["xyz"][act[:, s], s].mean(0))

    # ---- summary table
    print(f"   {'ch':>2} | {'BASELINE: active%':>17} {'az/el':>12} {'dB':>7} {'class':>6} | "
          f"{'WIENER: active%':>15} {'az/el':>12} {'dB':>7} {'class':>6}")
    nch = max((s.shape[1] for _, s in out.values() if s is not None), default=0)
    for c in range(nch):
        line = f"   {c:>2} |"
        for ver in ("raw", "filt"):
            tr, sep = out[ver]
            if tr is None or c >= tr["ids"].shape[1]:
                line += f" {'-':>17} {'-':>12} {'-':>7} {'-':>6} |"
                continue
            m = tr["ids"][:, c] != 0
            e = 10 * np.log10(np.mean(sep[:, c] ** 2) + 1e-20) if sep is not None and c < sep.shape[1] else np.nan
            if m.any():
                v = unit(tr["xyz"][m, c].mean(0))
                az, el = az_el(v, a.vertical)
                cls = "-" if known_ref is None else ("KNOWN" if np.mean(ang(tr["xyz"][m, c], known_ref) < 20) > .5
                                                     else "other")
                d = f"{float(az):5.0f}/{float(el):4.0f}"
            else:
                d, cls = "-", "-"
            w = 17 if ver == "raw" else 15
            line += f" {m.mean()*100:>{w}.0f} {d:>12} {e:>7.1f} {cls:>6} |"
        print(line.rstrip("|"))

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.5), sharey=True)
    for k, ver in enumerate(("raw", "filt")):
        tr = out[ver][0]
        if tr is not None:
            az, _ = az_el(tr["xyz"], a.vertical)
            t = np.arange(len(az)) * shop / sfs
            for s in range(az.shape[1]):
                m = tr["ids"][:, s] != 0
                ax[k].plot(t[m], az[m, s], ".", ms=2.5, label=f"ch {s}")
        if known_ref is not None:
            ax[k].axhline(float(az_el(known_ref, a.vertical)[0]), color="r", ls="--", lw=1, label="known")
        ax[k].set_title(f"{name} - SST {'BASELINE' if ver == 'raw' else 'WIENER before SSL'}")
        ax[k].set_xlabel("s"); ax[k].set_ylim(-180, 180); ax[k].grid(alpha=.3)
    ax[0].set_ylabel("horizontal angle (deg)"); ax[1].legend(fontsize=7, markerscale=3)
    fig.tight_layout(); fig.savefig(os.path.join(sd, "tracks.png"), dpi=105); plt.close(fig)

    seps = {v: s for v, (_, s) in out.items() if s is not None}
    if seps:
        vmax = max(10 * np.log10(np.max(s ** 2) + 1e-20) for s in seps.values())
        fig, ax = plt.subplots(nch, 2, figsize=(13, 2.1 * nch), sharex=True, sharey=True, squeeze=False)
        for j, ver in enumerate(("raw", "filt")):
            for c in range(nch):
                s = seps.get(ver)
                if s is not None and c < s.shape[1] and np.any(s[:, c]):
                    ax[c, j].specgram(s[:, c] + 1e-9, NFFT=1024, Fs=sfs, noverlap=512, cmap="magma",
                                      vmin=vmax - 110, vmax=vmax - 20)
                else:
                    ax[c, j].text(.5, .5, "silent", ha="center", va="center", transform=ax[c, j].transAxes)
                ax[c, j].set_ylim(0, 6000)
                if j == 0:
                    ax[c, j].set_ylabel(f"SSS ch {c}\nHz")
            ax[0, j].set_title(f"{name} - SSS {'BASELINE' if ver == 'raw' else 'WIENER before SSL'}")
        fig.tight_layout(); fig.savefig(os.path.join(sd, "sss_spectrograms.png"), dpi=100); plt.close(fig)
    return noise, known_ref


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--odas", default=os.path.expanduser("~/odas_ws/odas/build/bin/odaslive"))
    ap.add_argument("--vertical", action="store_true")
    ap.add_argument("--from", dest="src", help="reuse takes from this folder instead of recording")
    ap.add_argument("--scenes", default="calib,known,target,both,cross")
    ap.add_argument("--dur", type=float, default=30)
    ap.add_argument("--device", default="hw:2,0")
    ap.add_argument("--out", default="~/dataset/sss_tests")
    ap.add_argument("--frame", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--fmin", type=float, default=180)
    ap.add_argument("--fmax", type=float, default=3600)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=0.98)
    ap.add_argument("--full-band", dest="full_band", action="store_true")
    a = ap.parse_args()
    a.odas = os.path.expanduser(a.odas)
    if not os.access(a.odas, os.X_OK):
        sys.exit(f"[FAIL] odaslive not found at {a.odas}")
    p = dict(frame=a.frame, hop=a.hop, fmin=a.fmin, fmax=a.fmax, alpha=a.alpha, floor=a.floor,
             beta=a.beta, full_band=a.full_band)
    base_cfg = open(os.path.expanduser(a.cfg)).read()
    info = cfg_info(base_cfg)
    rfs = int(info["raw"]["fS"] or 44100)

    sess = os.path.join(os.path.abspath(os.path.expanduser(a.out)), datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sess)
    json.dump(dict(params=p, cfg=a.cfg, odas=a.odas, vertical=a.vertical),
              open(os.path.join(sess, "session.json"), "w"), indent=2)
    print(f"Session folder: {sess}\nWiener params: {p}")

    wanted = [s for s in a.scenes.split(",") if s]
    if "calib" not in wanted:
        wanted.insert(0, "calib")
    noise, known_ref = None, None
    for name, what in SCENES:
        if name not in wanted:
            continue
        sd = os.path.join(sess, name)
        os.makedirs(sd)
        print("\n" + "=" * 90 + f"\n {name.upper()}: {what}\n" + "=" * 90)
        if a.src:
            wav = find_take(os.path.expanduser(a.src), name)
            if not wav:
                print(f"   [skip] no *{name}*.wav in {a.src}")
                continue
            print(f"   using {wav}")
        else:
            input(f"   Set up the scene, then press ENTER to record {a.dur:.0f} s... ")
            wav = os.path.join(sd, "take.wav")
            record(wav, a.dur, a.device, rfs)
        noise, known_ref = process_scene(name, wav, sd, a, p, info, base_cfg, noise, known_ref)
        if os.path.basename(wav) == "take.wav":
            os.remove(wav)                       # identical to mics_raw.wav
        print(f"   saved -> {sd}/  (sss_raw_ch*.wav, sss_filt_ch*.wav, tracks*.csv, *.png)")

    print(f"\nDone. All WAVs: {sess}\n  ls {sess}/*/sss_*.wav")


if __name__ == "__main__":
    main()
