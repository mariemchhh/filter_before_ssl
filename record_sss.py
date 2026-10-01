#!/usr/bin/env python3
"""record_sss.py - record ODAS SSS outputs straight to .wav with odaslive only (no ROS).
Run folder contains only: separated_src0..N.wav, postfiltered_src0..N.wav (+ *_norm.wav with --normalize)
"""
import argparse, os, shutil, signal, subprocess, sys, tempfile, time, wave
import numpy as np
try:
    import libconf
except ImportError:
    sys.exit("missing dependency: pip install libconf")

DTYPES = {8: np.int8, 16: np.int16, 32: np.int32}
STREAMS = ("separated", "postfiltered")

def find_odaslive(user_path):
    if user_path:
        return os.path.expanduser(user_path)
    for cand in (shutil.which("odaslive"),
                 os.path.expanduser("~/odas_ws/odas/build/bin/odaslive"),
                 os.path.expanduser("~/odas/build/bin/odaslive")):
        if cand and os.path.isfile(cand):
            return cand
    sys.exit("odaslive not found - pass --odaslive /path/to/odaslive")

def patch_cfg(src_cfg, tmp_dir, streams):
    with open(os.path.expanduser(src_cfg)) as f:
        cfg = libconf.load(f)
    if cfg["raw"]["interface"]["type"] == "socket":
        sys.exit("raw input is a socket (odas_ros/feeder config) - use a soundcard config")
    info = {}
    for name in STREAMS:
        blk = cfg["sss"].get(name)
        if blk is None:
            continue
        if name in streams:
            path = os.path.join(tmp_dir, f"{name}.raw")
            blk["interface"] = {"type": "file", "path": path}
            info[name] = {"fs": int(blk["fS"]), "bits": int(blk["nBits"]), "path": path}
        else:
            blk["interface"] = {"type": "blackhole"}
    run_cfg = os.path.join(tmp_dir, "run.cfg")
    with open(run_cfg, "w") as f:
        libconf.dump(cfg, f)
    return run_cfg, info

def run_odas(odaslive, run_cfg, duration):
    proc = subprocess.Popen([odaslive, "-c", run_cfg])
    t0 = time.monotonic()
    try:
        while proc.poll() is None:
            if duration > 0 and time.monotonic() - t0 >= duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    return time.monotonic() - t0, proc.returncode

def write_wav(path, data, fs, width):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(width); w.setframerate(fs)
        w.writeframes(data.tobytes())

def to_wavs(name, meta, nch, run_dir, normalize):
    bits, fs = meta["bits"], meta["fs"]
    if bits not in DTYPES:
        print(f"[{name}] nBits={bits} unsupported"); return
    if not os.path.isfile(meta["path"]) or os.path.getsize(meta["path"]) == 0:
        print(f"[{name}] no data written"); return
    x = np.fromfile(meta["path"], dtype=DTYPES[bits])
    x = x[:(len(x) // nch) * nch].reshape(-1, nch)
    full = float(2 ** (bits - 1))
    for c in range(nch):
        ch = np.ascontiguousarray(x[:, c])
        base = os.path.join(run_dir, f"{name}_src{c}")
        write_wav(base + ".wav", ch, fs, bits // 8)
        f = ch.astype(np.float64) / full
        rms = 20 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-12)
        peak = np.max(np.abs(f)) if len(f) else 0.0
        msg = f"{name}_src{c}.wav  {len(f)/fs:5.1f} s  rms {rms:6.1f} dBFS  peak {20*np.log10(peak+1e-12):6.1f} dBFS"
        if normalize and peak > 0:
            y = np.clip(f * (10 ** (-3 / 20) / peak), -1, 1)
            write_wav(base + "_norm.wav", (y * 32767).astype(np.int16), fs, 2)
            msg += "  (+ _norm)"
        print(msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds, 0 = until Ctrl+C")
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--outroot", default="~/sss_runs")
    ap.add_argument("--odaslive", default=None)
    ap.add_argument("--no-postfiltered", action="store_true")
    ap.add_argument("--normalize", action="store_true",
                    help="also write *_norm.wav scaled to -3 dBFS peak (for listening)")
    a = ap.parse_args()

    run_dir = os.path.join(os.path.expanduser(a.outroot),
                           f"{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir)
    tmp_dir = tempfile.mkdtemp(prefix="odas_sss_")
    try:
        streams = ("separated",) if a.no_postfiltered else STREAMS
        run_cfg, info = patch_cfg(a.config, tmp_dir, streams)
        odaslive = find_odaslive(a.odaslive)
        print(f"run dir : {run_dir}\nodaslive: {odaslive}")
        print(f"recording {'until Ctrl+C' if a.duration <= 0 else f'{a.duration:.0f} s'} - odas_web must already be running")
        elapsed, rc = run_odas(odaslive, run_cfg, a.duration)
        print(f"odaslive stopped after {elapsed:.1f} s (rc={rc})\n")
        for name, m in info.items():
            to_wavs(name, m, a.channels, run_dir, a.normalize)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
