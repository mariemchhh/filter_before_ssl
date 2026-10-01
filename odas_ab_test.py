#!/usr/bin/env python3
"""
odas_ab_test.py - Run ODAS on the SAME scene twice: unfiltered (baseline) and Wiener-filtered
(before SSL), then compare what SST tracks and what the SSS channels sound like.

  source ~/ros2_ws/install/setup.bash          # only needed if ODAS runs through ros2
  python3 odas_ab_test.py --cfg ~/Downloads/uma16.cfg --vertical
  xdg-open ~/dataset/odas_AB/report.html

Inputs (defaults):
  ~/dataset/S*.wav                       raw 16-ch takes
  ~/dataset/analysis_A/S*_filtA.wav      Wiener-filtered takes (from wiener_analysis.py)

For each scenario and each version (raw / filt) it:
  1. writes the 16-ch input as headerless PCM in the cfg's nBits format
  2. writes a cfg copy whose interfaces are files: raw input, potential, tracked, separated, postfiltered
  3. runs ODAS (odaslive, or odas_core_node via ros2 - autodetected; override with --odas)
  4. parses tracked (SST) and separated (SSS, one channel per track slot)
Then writes ~/dataset/odas_AB/report.html with:
  - SST tracks over time (raw vs filt), with the known / target reference directions
  - per SSS channel: active %, direction, class (KNOWN / TARGET / other), energy, players before/after
  - SSS spectrograms before/after
Nothing goes through the sound card: ODAS reads files, so PulseAudio / device busy do not matter.
"""
import argparse
import glob
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINKS = {"potential": "potential.json", "tracked": "tracked.json",
         "separated": "separated.raw", "postfiltered": "postfiltered.raw"}
NEAR_DEG = 20.0


# ================================================================ cfg
def _find(txt, key):
    m = re.search(r"\b%s\s*:\s*\{" % key, txt)
    if not m:
        return None
    im = re.compile(r"interface\s*:\s*\{[^{}]*\}\s*;?").search(txt, m.end())
    return (m, im) if im else None


def _val(seg, key):
    m = re.search(r"\b%s\s*=\s*([-\d.]+)" % key, seg)
    return float(m.group(1)) if m else None


def cfg_info(txt):
    info = {}
    for key in ["raw"] + list(SINKS):
        r = _find(txt, key)
        if r:
            m, im = r
            seg = txt[m.end():im.start()]
            info[key] = {k: _val(seg, k) for k in ("fS", "hopSize", "nBits", "nChannels")}
    for k in ("raw", "tracked", "separated"):
        if k not in info:
            sys.exit(f"[FAIL] cfg: section '{k}' with an interface block not found")
    return info


def make_cfg(txt, raw_in, outdir):
    for key in ["raw"] + list(SINKS):
        r = _find(txt, key)
        if not r:
            continue
        _, im = r
        path = raw_in if key == "raw" else os.path.join(outdir, SINKS[key])
        txt = txt[:im.start()] + f'interface: {{ type = "file"; path = "{path}"; }};' + txt[im.end():]
    return txt


# ================================================================ ODAS
def detect_odas():
    p = shutil.which("odaslive")
    if p:
        return p + " -c {cfg}"
    for root in ("~/ros2_ws", "~/odas", "~/odas_ros"):
        for c in glob.glob(os.path.expanduser(root + "/**/odaslive"), recursive=True):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c + " -c {cfg}"
    if shutil.which("ros2"):
        return "ros2 run odas_ros odas_core_node --ros-args -p configuration_path:={cfg}"
    sys.exit("[FAIL] no ODAS found. Source ROS 2 (source ~/ros2_ws/install/setup.bash) "
             "or pass --odas 'path/to/odaslive -c {cfg}'")


def run_odas(cmd_t, cfg, timeout, log):
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd_t.format(cfg=cfg), shell=True, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            p.wait(timeout=timeout)
            return "ended"
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGINT, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(2)
            return "stopped after timeout"


# ================================================================ I/O
def write_pcm(path, x, nbits):
    if nbits == 16:
        (np.clip(x, -1, 1 - 2 ** -15) * 2 ** 15).astype("<i2").tofile(path)
    elif nbits == 24:
        sys.exit("[FAIL] nBits = 24 in cfg raw: use 32 (S32_LE) or 16")
    else:
        (np.clip(x, -1, 1 - 2 ** -31) * 2 ** 31).astype("<i4").tofile(path)


def read_tracked(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    dec, i, frames = json.JSONDecoder(), 0, []
    while True:
        i = txt.find("{", i)
        if i < 0:
            break
        try:
            obj, i = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and "src" in obj:
            frames.append(obj)
    if not frames:
        return None
    S = max(len(f["src"]) for f in frames)
    T = len(frames)
    ids = np.zeros((T, S), int)
    xyz = np.zeros((T, S, 3))
    act = np.zeros((T, S))
    for t, fr in enumerate(frames):
        for s, src in enumerate(fr["src"]):
            ids[t, s] = int(src.get("id", 0))
            xyz[t, s] = [src.get("x", 0.0), src.get("y", 0.0), src.get("z", 0.0)]
            act[t, s] = float(src.get("activity", 1.0 if src.get("id", 0) else 0.0))
    return dict(ids=ids, xyz=xyz, act=act)


def read_sep(path, nbits, nch):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = np.fromfile(path, dtype="<i2" if nbits == 16 else "<i4")
    n = len(d) // nch * nch
    return d[:n].reshape(-1, nch) / float(2 ** (nbits - 1))


# ================================================================ geometry
def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def az_el(xyz, vertical):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if vertical:   # board vertical: x horizontal, y vertical, z depth
        return np.degrees(np.arctan2(x, z)), np.degrees(np.arctan2(y, np.hypot(x, z)))
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.hypot(x, y)))


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def main_dir(tr):
    """Mean direction of the most active slot."""
    if tr is None:
        return None
    active = tr["ids"] != 0
    s = int(np.argmax(active.mean(0)))
    if not active[:, s].any():
        return None
    return unit(tr["xyz"][active[:, s], s].mean(0))


# ================================================================ analysis
def analyse(tr, sep, refs, vertical):
    T, S = tr["ids"].shape
    active = tr["ids"] != 0
    res = dict(frames=T, any_track=float(active.any(1).mean() * 100), slots=[])
    for name, ref in refs.items():
        if ref is None:
            res[f"near_{name}"] = None
            continue
        near = active & (ang(tr["xyz"], ref) < NEAR_DEG)
        res[f"near_{name}"] = float(near.any(1).mean() * 100)
    for s in range(S):
        a = active[:, s]
        d = dict(slot=s, active=float(a.mean() * 100), az=None, el=None, cls="-", ids="-", energy=None)
        if a.any():
            v = tr["xyz"][a, s]
            az, el = az_el(unit(v.mean(0)), vertical)
            d["az"], d["el"] = float(az), float(el)
            d["ids"] = ",".join(str(i) for i in sorted(set(tr["ids"][a, s].tolist())))
            votes = {}
            for name, ref in refs.items():
                if ref is not None:
                    votes[name] = float(np.mean(ang(v, ref) < NEAR_DEG))
            best = max(votes, key=votes.get) if votes else None
            d["cls"] = best.upper() if best and votes[best] > 0.5 else "other"
        if sep is not None and s < sep.shape[1]:
            d["energy"] = float(10 * np.log10(np.mean(sep[:, s] ** 2) + 1e-20))
        res["slots"].append(d)
    return res


# ================================================================ plots
def plot_tracks(runs, refs, vertical, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for a, (ver, (tr, hop_s)) in zip(ax, runs.items()):
        if tr is not None:
            T, S = tr["ids"].shape
            t = np.arange(T) * hop_s
            az, _ = az_el(tr["xyz"], vertical)
            for s in range(S):
                m = tr["ids"][:, s] != 0
                a.plot(t[m], az[m, s], ".", ms=2.5, label=f"slot {s}")
        for name, ref, c in (("known", refs.get("known"), "r"), ("target", refs.get("target"), "g")):
            if ref is not None:
                a.axhline(float(az_el(ref, vertical)[0]), color=c, ls="--", lw=1, label=f"{name} ref")
        a.set_title(f"{title} - {'BASELINE (no filter)' if ver == 'raw' else 'WIENER before SSL'}")
        a.set_xlabel("s"); a.grid(alpha=.3); a.set_ylim(-180, 180)
    ax[0].set_ylabel("horizontal angle (deg)")
    ax[1].legend(fontsize=7, loc="upper right", markerscale=3)
    fig.tight_layout(); fig.savefig(path, dpi=105); plt.close(fig)


def plot_sss(seps, fs, path, title):
    nch = max(s.shape[1] for s in seps.values() if s is not None)
    fig, ax = plt.subplots(nch, 2, figsize=(13, 2.1 * nch), sharex=True, sharey=True, squeeze=False)
    vmax = max(10 * np.log10(np.max(s ** 2) + 1e-20) for s in seps.values() if s is not None)
    for j, ver in enumerate(("raw", "filt")):
        s = seps.get(ver)
        for c in range(nch):
            a = ax[c, j]
            if s is None or c >= s.shape[1] or not np.any(s[:, c]):
                a.text(0.5, 0.5, "silent / no data", ha="center", va="center", transform=a.transAxes)
            else:
                a.specgram(s[:, c] + 1e-9, NFFT=1024, Fs=fs, noverlap=512, cmap="magma",
                           vmin=vmax - 110, vmax=vmax - 20)
            a.set_ylim(0, 6000)
            if c == 0:
                a.set_title(f"{title} - SSS {'BASELINE' if ver == 'raw' else 'WIENER before SSL'}")
            if j == 0:
                a.set_ylabel(f"ch {c}\nHz")
    ax[-1, 0].set_xlabel("s"); ax[-1, 1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dataset", default="~/dataset")
    ap.add_argument("--filtered", default="~/dataset/analysis_A", help="folder with S*_filtA.wav")
    ap.add_argument("--out", default="~/dataset/odas_AB")
    ap.add_argument("--odas", help="command template, e.g. '~/odas/build/bin/odaslive -c {cfg}'")
    ap.add_argument("--vertical", action="store_true", help="board mounted vertically (z = depth)")
    ap.add_argument("--extra", type=float, default=30.0, help="seconds added to the ODAS timeout")
    a = ap.parse_args()
    ds, fd, out = (os.path.abspath(os.path.expanduser(p)) for p in (a.dataset, a.filtered, a.out))
    os.makedirs(out, exist_ok=True)
    listen = os.path.join(out, "listen")
    os.makedirs(listen, exist_ok=True)

    base = open(os.path.expanduser(a.cfg)).read()
    info = cfg_info(base)
    rb = int(info["raw"]["nBits"] or 32)
    rch = int(info["raw"]["nChannels"] or 16)
    rfs = int(info["raw"]["fS"] or 44100)
    sb = int(info["separated"]["nBits"] or 16)
    sfs = int(info["separated"]["fS"] or rfs)
    shop = int(info["separated"]["hopSize"] or info["raw"]["hopSize"] or 512)
    cmd = a.odas or detect_odas()
    print(f"ODAS command : {cmd}")
    print(f"cfg raw      : {rch} ch, {rfs} Hz, {rb} bit | separated: {sfs} Hz, {sb} bit, hop {shop}")

    scen = []
    for w in sorted(glob.glob(os.path.join(ds, "S*.wav"))):
        stem = os.path.splitext(os.path.basename(w))[0]
        f = os.path.join(fd, stem + "_filtA.wav")
        if os.path.exists(f):
            scen.append((stem, w, f))
    if not scen:
        sys.exit(f"[FAIL] no S*.wav in {ds} with a matching *_filtA.wav in {fd}")

    # ---------------- run ODAS
    data = {}
    for stem, wraw, wfilt in scen:
        for ver, wav in (("raw", wraw), ("filt", wfilt)):
            d = os.path.join(out, stem, ver)
            os.makedirs(d, exist_ok=True)
            for f in SINKS.values():
                if os.path.exists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
            x, sr = sf.read(wav, dtype="float64", always_2d=True)
            if sr != rfs or x.shape[1] != rch:
                sys.exit(f"[FAIL] {wav}: {x.shape[1]} ch / {sr} Hz but cfg raw expects {rch} ch / {rfs} Hz")
            pcm = os.path.join(d, "input.raw")
            write_pcm(pcm, x, rb)
            cfgp = os.path.join(d, "odas_file.cfg")
            open(cfgp, "w").write(make_cfg(base, pcm, d))
            t0 = time.time()
            how = run_odas(cmd, cfgp, len(x) / sr + a.extra, os.path.join(d, "odas.log"))
            tr = read_tracked(os.path.join(d, SINKS["tracked"]))
            nch = tr["ids"].shape[1] if tr else 4
            sep = read_sep(os.path.join(d, SINKS["separated"]), sb, nch)
            data[(stem, ver)] = (tr, sep)
            st = "OK" if tr is not None and sep is not None else "NO OUTPUT"
            print(f"[{st:9s}] {stem:10s} {ver:4s}: ODAS {how} in {time.time()-t0:5.1f} s | "
                  f"SST frames {tr['ids'].shape[0] if tr else 0} | "
                  f"SSS {sep.shape if sep is not None else None}  (log: {d}/odas.log)")
            if st != "OK":
                print("            -> read the log; check that ODAS accepts type = \"file\" interfaces")

    # ---------------- reference directions from the baselines
    refs = {"known": None, "target": None}
    for key, words in (("known", ("known", "calib")), ("target", ("target",))):
        for w in words:                       # S1_known preferred, S0_calib as fallback
            for stem, _, _ in scen:
                if refs[key] is None and w in stem:
                    refs[key] = main_dir(data[(stem, "raw")][0])
    for k, v in refs.items():
        if v is not None:
            az, el = az_el(v, a.vertical)
            print(f"reference {k:6s}: az {float(az):6.1f} deg, el {float(el):5.1f} deg")
        else:
            print(f"reference {k:6s}: not found (needs S1_known / S2_target baseline tracks)")
    if refs["known"] is not None and refs["target"] is not None:
        print(f"known-target separation: {float(ang(refs['known'], refs['target'])):.1f} deg")

    # ---------------- analysis + report
    H = ["<!doctype html><html><head><meta charset='utf-8'><title>ODAS raw vs Wiener</title><style>"
         "body{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:16px}"
         "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:.9em}"
         "th{background:#eee}img{max-width:100%}audio{width:220px}.KNOWN{color:#c00;font-weight:bold}"
         ".TARGET{color:#080;font-weight:bold}h2{border-bottom:2px solid #ccc;padding-top:14px}</style>"
         "</head><body><h1>ODAS on raw vs Wiener-filtered input (before SSL)</h1>",
         f"<p>cfg: {html.escape(a.cfg)} | ODAS: <code>{html.escape(cmd)}</code> | "
         f"a track is 'near' a reference if within {NEAR_DEG:.0f}&deg;.</p>"]
    summary = []
    for stem, _, _ in scen:
        hop_s = shop / sfs
        runs, seps, res = {}, {}, {}
        for ver in ("raw", "filt"):
            tr, sep = data[(stem, ver)]
            runs[ver] = (tr, hop_s)
            seps[ver] = sep
            res[ver] = analyse(tr, sep, refs, a.vertical) if tr is not None else None
        for ver in ("raw", "filt"):
            r = res[ver]
            if r:
                summary.append([stem, "baseline" if ver == "raw" else "Wiener",
                                f"{r['any_track']:.0f}",
                                "-" if r["near_known"] is None else f"{r['near_known']:.0f}",
                                "-" if r["near_target"] is None else f"{r['near_target']:.0f}"])
        trk = os.path.join(out, f"{stem}_tracks.png")
        spc = os.path.join(out, f"{stem}_sss_spec.png")
        plot_tracks(runs, refs, a.vertical, trk, stem)
        if any(s is not None for s in seps.values()):
            plot_sss(seps, sfs, spc, stem)

        # listening files: one common gain per scenario (levels comparable)
        peak = max((np.max(np.abs(s)) for s in seps.values() if s is not None and s.size), default=0)
        g = 0.9 / peak if peak > 0 else 1.0
        for ver, s in seps.items():
            if s is None:
                continue
            for c in range(s.shape[1]):
                sf.write(os.path.join(listen, f"{stem}_{ver}_ch{c}.wav"), np.clip(s[:, c] * g, -1, 1),
                         sfs, subtype="PCM_16")

        H.append(f"<h2>{stem}</h2><img src='{os.path.basename(trk)}'>")
        H.append("<table><tr><th>SSS ch</th>"
                 "<th>BASELINE: active %</th><th>dir az/el</th><th>class</th><th>dB</th><th>listen</th>"
                 "<th>WIENER: active %</th><th>dir az/el</th><th>class</th><th>dB</th><th>listen</th></tr>")
        nch = max((len(r["slots"]) for r in res.values() if r), default=0)
        for c in range(nch):
            row = f"<tr><td>{c}</td>"
            for ver in ("raw", "filt"):
                r = res[ver]
                if not r or c >= len(r["slots"]):
                    row += "<td colspan=5>-</td>"
                    continue
                d = r["slots"][c]
                dr = "-" if d["az"] is None else f"{d['az']:.0f}&deg; / {d['el']:.0f}&deg;"
                en = "-" if d["energy"] is None else f"{d['energy']:.1f}"
                wav = f"listen/{stem}_{ver}_ch{c}.wav"
                pl = f"<audio controls src='{wav}'></audio>" if os.path.exists(os.path.join(out, wav)) else "-"
                row += (f"<td>{d['active']:.0f}</td><td>{dr}</td><td class='{d['cls']}'>{d['cls']}</td>"
                        f"<td>{en}</td><td>{pl}</td>")
            H.append(row + "</tr>")
        H.append("</table>")
        if os.path.exists(spc):
            H.append(f"<img src='{os.path.basename(spc)}'>")

    H.insert(2, "<h2>Summary</h2><table><tr><th>scenario</th><th>input</th><th>frames with a track %</th>"
                "<th>frames with a track ON KNOWN %</th><th>frames with a track ON TARGET %</th></tr>" +
             "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in summary) +
             "</table><p><b>Goal:</b> with Wiener, 'on known' drops (ideally to ~0 in S1/S3) "
             "and 'on target' stays the same or rises (S2, S3, and especially S4).</p>")
    H.append("</body></html>")
    rep = os.path.join(out, "report.html")
    open(rep, "w").write("\n".join(H))

    print("\nscenario    input     track%  on-known%  on-target%")
    for r in summary:
        print(f"{r[0]:11s} {r[1]:9s} {r[2]:>6s}  {r[3]:>9s}  {r[4]:>10s}")
    print(f"\nReport -> {rep}")


if __name__ == "__main__":
    main()
