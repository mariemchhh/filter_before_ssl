#!/usr/bin/env python3
"""
wiener_prefilter.py - Wiener filter on the 16 UMA-16 mics, BEFORE ODAS (SSL + SSS).

The known source (fixed position, fixed energy) is learned once, then removed
from all 16 channels with ONE common gain per frequency bin, so the phase
differences between mics (= the directions) are untouched.

Commands
  profile  learn the known source from a 16-ch recording where only it is active
             python3 wiener_prefilter.py profile ~/dataset/S0_calib.wav -o ~/wiener_profile.npz

  apply    filter a 16-ch recording offline -> 16-ch WAV + .raw for ODAS file input
             python3 wiener_prefilter.py apply ~/dataset/S3_both.wav -p ~/wiener_profile.npz

  check    compare SRP-PHAT directions before/after: known source should vanish,
           other directions must not move
             python3 wiener_prefilter.py check ~/dataset/S3_both.wav ~/dataset/S3_both_wiener.wav \
                 -p ~/wiener_profile.npz

  Tuning for SSL: SSL uses PHAT (phase only), so what matters is that the
  known source no longer OWNS its bins. Use --floor 0 (default). --over raises
  the removal strength. --mask (bins below it set to 0) is experimental: in
  tests it removed too many target bins and degraded the target direction.

  stream   real time: UMA-16 (arecord) -> Wiener -> TCP -> ODAS (raw socket input)
             python3 wiener_prefilter.py stream -p ~/wiener_profile.npz --port 9200
           start it BEFORE odas_core_node (ODAS connects to it as a client)

Record 16-ch input with ODAS stopped:
    arecord -D hw:2,0 -c 16 -r 44100 -f S32_LE -d 30 ~/dataset/S0_calib.wav
"""

import argparse
import socket
import subprocess
import sys
import time
import wave

import numpy as np

FS, NFFT, HOP, NCH, C = 44100, 1024, 512, 16, 343.0
WIN = np.sqrt(np.hanning(NFFT + 1)[:-1])          # sqrt periodic Hann: perfect OLA at 50 %
F = np.fft.rfftfreq(NFFT, 1 / FS)

# UMA-16 positions from uma16.cfg, MIC1..MIC16 (metres)
MICS = np.array([
    (-0.021, -0.063), (-0.063, -0.063), (-0.021, -0.021), (-0.063, -0.021),
    (-0.021, +0.021), (-0.063, +0.021), (-0.021, +0.063), (-0.063, +0.063),
    (+0.063, +0.063), (+0.021, +0.063), (+0.063, +0.021), (+0.021, +0.021),
    (+0.063, -0.021), (+0.021, -0.021), (+0.063, -0.063), (+0.021, -0.063)])
MICS = np.column_stack([MICS, np.zeros(NCH)])


# ----------------------------------------------------------------------------- io
def read_wav16(path):
    with wave.open(path) as w:
        ch, sw, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if ch != NCH or fs != FS:
        sys.exit(f'{path}: need {NCH} ch @ {FS} Hz (got {ch} ch @ {fs} Hz)')
    if sw not in (2, 4):
        sys.exit(f'{path}: {8 * sw}-bit not supported (use S16_LE or S32_LE)')
    x = np.frombuffer(raw, {2: np.int16, 4: np.int32}[sw]).reshape(-1, NCH)
    return x.astype(np.float64) / 2 ** (8 * sw - 1), sw


def to_int(x, sw):
    full = 2 ** (8 * sw - 1)
    return np.clip(np.round(x * full), -full, full - 1).astype({2: np.int16, 4: np.int32}[sw])


def write_wav16(path, x, sw):
    with wave.open(path, 'wb') as w:
        w.setnchannels(NCH)
        w.setsampwidth(sw)
        w.setframerate(FS)
        w.writeframes(to_int(x, sw).tobytes())


# ----------------------------------------------------------------------------- core
class StreamWiener:
    """Hop-by-hop Wiener filter, common gain for all mics. Latency = 1 hop (11.6 ms)."""

    def __init__(self, noise_psd, over=1.5, floor=0.0, dd=0.95, mask=0.0):
        self.N = over * noise_psd + 1e-20
        self.floor, self.dd, self.mask = floor, dd, mask
        self.prev = np.zeros((NCH, HOP))
        self.ola = np.zeros((NCH, HOP))
        self.prev_S = np.zeros(len(F))
        self.last_gain = np.ones(len(F))

    def process(self, hop):
        """hop: (HOP, NCH) float -> (HOP, NCH) float, delayed by one hop."""
        frame = np.concatenate([self.prev, hop.T], axis=1)        # (NCH, NFFT)
        self.prev = hop.T.copy()
        X = np.fft.rfft(frame * WIN, axis=1)                      # (NCH, F)
        P = np.mean(np.abs(X) ** 2, axis=0)                       # common power over mics
        gamma = P / self.N
        xi = self.dd * self.prev_S / self.N + (1 - self.dd) * np.maximum(gamma - 1, 0)
        G = np.maximum(xi / (1 + xi), self.floor)
        self.prev_S = G ** 2 * P
        if self.mask > 0:
            G = np.where(G < self.mask, 0.0, G)                   # bins owned by the known source -> exactly 0
        self.last_gain = G
        y = np.fft.irfft(X * G, n=NFFT, axis=1) * WIN             # same G on every mic
        out = self.ola + y[:, :HOP]
        self.ola = y[:, HOP:]
        return out.T


def stft_all(x):                                                   # x: (n, NCH) -> (NCH, T, F)
    n = (len(x) - NFFT) // HOP + 1
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x.T[:, idx] * WIN, axis=-1)


# ----------------------------------------------------------------------------- SRP-PHAT (check)
def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


AZ_G, EL_G = np.meshgrid(np.arange(-180, 180, 5), np.arange(0, 91, 5))
U_G = unit(AZ_G.ravel(), EL_G.ravel())
BAND = (F >= 180) & (F <= 3600)
STEER = np.exp(2j * np.pi * (-(U_G @ MICS.T) / C)[..., None] * F[BAND])   # conj steering (G, M, Fb)


def srp_peaks(X, every=4, n=2, excl=40):
    """Top-n SRP-PHAT directions every `every` frames -> list of [(unit_vec, power), ...]."""
    out = []
    for t in range(0, X.shape[1], every):
        Y = X[:, t, BAND]
        Y = Y / (np.abs(Y) + 1e-12)
        p = (np.abs(np.einsum('gmf,mf->gf', STEER, Y)) ** 2).sum(-1)
        pk = []
        for _ in range(n):
            i = int(np.argmax(p))
            pk.append((U_G[i], float(p[i])))
            p = np.where(np.degrees(np.arccos(np.clip(U_G @ U_G[i], -1, 1))) < excl, 0, p)
        out.append(pk)
    return out


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


def az_el(u):
    return np.degrees(np.arctan2(u[1], u[0])), np.degrees(np.arcsin(np.clip(u[2], -1, 1)))


# ----------------------------------------------------------------------------- commands
def cmd_profile(a):
    x, _ = read_wav16(a.wav)
    X = stft_all(x)
    P = np.mean(np.abs(X) ** 2, axis=0)                           # (T, F) mean over mics
    e = 10 * np.log10(P.mean(axis=1) + 1e-20)
    keep = e >= np.percentile(e, a.keep_pct)
    noise = np.median(P[keep], axis=0)
    spread = float(1.4826 * np.median(np.abs(e[keep] - np.median(e[keep]))))

    pk = srp_peaks(X[:, keep], every=4, n=1)
    V = np.array([p[0][0] for p in pk])
    kd = V.mean(0)
    kd /= np.linalg.norm(kd)
    kaz, kel = az_el(kd)

    np.savez(a.out, noise_psd=noise, known_dir=kd, level_db=float(np.median(e[keep])),
             spread_db=spread, fs=FS)
    top = np.argsort(noise * BAND)[::-1][:5]
    print(f'learned from {keep.sum()} frames, level {np.median(e[keep]):.1f} dB (spread {spread:.2f} dB)')
    print(f'known direction (SRP-PHAT)  az {kaz:.0f} deg  el {kel:.0f} deg')
    print('strongest bins: ' + ', '.join(f'{F[i]:.0f} Hz' for i in sorted(top)))
    if spread > 3:
        print('WARNING: level spread > 3 dB - the known source is not as stable as assumed')
    print(f'profile -> {a.out}')


def cmd_apply(a):
    P = np.load(a.profile)
    x, sw = read_wav16(a.wav)
    wf = StreamWiener(P['noise_psd'], a.over, a.floor, a.dd, a.mask)
    n = len(x) // HOP * HOP
    xp = np.concatenate([x[:n], np.zeros((HOP, NCH))])              # flush the 1-hop latency
    y = np.concatenate([wf.process(xp[i:i + HOP]) for i in range(0, len(xp), HOP)])[HOP:HOP + n]

    base = a.out or a.wav.rsplit('.', 1)[0] + '_wiener'
    write_wav16(base + '.wav', y, sw)
    to_int(y, sw).tofile(base + '.raw')

    Xb, Xa = stft_all(x[:n]), stft_all(y)
    pb = np.mean(np.abs(Xb) ** 2, axis=0) * BAND
    pa = np.mean(np.abs(Xa) ** 2, axis=0) * BAND
    red = 10 * np.log10(pb.sum() / (pa.sum() + 1e-20))
    print(f'{n / FS:.1f} s, {8 * sw}-bit, over={a.over} floor={a.floor} dd={a.dd} mask={a.mask}')
    print(f'in-band energy reduction {red:.1f} dB (all sources, 180-3600 Hz)')
    print(f'-> {base}.wav   (listen / inspect)')
    print(f'-> {base}.raw   (ODAS raw input: type = "file"; nBits = {8 * sw})')


def cmd_check(a):
    P = np.load(a.profile)
    kd = P['known_dir']
    xb, _ = read_wav16(a.before)
    xa, _ = read_wav16(a.after)
    n = min(len(xb), len(xa))
    pb = srp_peaks(stft_all(xb[:n]))
    pa = srp_peaks(stft_all(xa[:n]))

    def known_top1(pk):
        return 100 * np.mean([ang(f[0][0], kd) <= a.tol for f in pk])

    def known_margin(pk):
        m = []
        for f in pk:
            kp = [pw for u, pw in f if ang(u, kd) <= a.tol]
            op = [pw for u, pw in f if ang(u, kd) > a.tol]
            if kp and op:
                m.append(10 * np.log10(kp[0] / op[0]))
        return np.median(m) if m else float('nan')

    shift = []
    for fb, fa in zip(pb, pa):
        ob = [u for u, _ in fb if ang(u, kd) > a.tol]
        oa = [u for u, _ in fa if ang(u, kd) > a.tol]
        if ob and oa:
            shift.append(ang(ob[0], oa[0]))                       # strongest other source
    kaz, kel = az_el(kd)
    print(f'known direction az {kaz:.0f} el {kel:.0f}, tolerance {a.tol} deg, {len(pb)} frames checked')
    print(f'known source = strongest SSL peak:  before {known_top1(pb):5.1f} %   after {known_top1(pa):5.1f} %')
    print(f'known peak vs other peak (median):  before {known_margin(pb):+5.1f} dB  after {known_margin(pa):+5.1f} dB')
    if shift:
        s = np.array(shift)
        print(f'target direction after vs before:   median shift {np.median(s):.1f} deg, '
              f'90 % within {np.percentile(s, 90):.1f} deg')
        ok = np.median(s) <= 7.5
        print('PHASE CHECK ' + ('OK - directions preserved' if ok else 'FAILED - directions moved'))
    else:
        print('no other direction found after filtering (only the known source in this take?)')


def cmd_stream(a):
    P = np.load(a.profile)
    wf = StreamWiener(P['noise_psd'], a.over, a.floor, a.dd, a.mask)
    bps = 4
    hop_bytes = HOP * NCH * bps

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', a.port))
    srv.listen(1)
    print(f'listening on 127.0.0.1:{a.port} - start odas_core_node now (raw interface = socket)')
    conn, _ = srv.accept()
    print('ODAS connected - capturing from ' + a.device)

    rec = subprocess.Popen(['arecord', '-q', '-D', a.device, '-c', str(NCH), '-r', str(FS),
                            '-f', 'S32_LE', '-t', 'raw'], stdout=subprocess.PIPE)
    hops, t_proc, t0 = 0, 0.0, time.monotonic()
    try:
        while True:
            buf = rec.stdout.read(hop_bytes)
            if len(buf) < hop_bytes:
                print('capture ended')
                break
            x = np.frombuffer(buf, np.int32).reshape(HOP, NCH).astype(np.float64) / 2 ** 31
            t1 = time.perf_counter()
            y = wf.process(x)
            t_proc += time.perf_counter() - t1
            conn.sendall(to_int(y, 4).tobytes())
            hops += 1
            if hops % 860 == 0:
                print(f'{hops} hops ({time.monotonic() - t0:.0f} s), '
                      f'filter {1000 * t_proc / hops:.2f} ms/hop (budget 11.6 ms)')
    except (BrokenPipeError, ConnectionResetError):
        print('ODAS disconnected')
    except KeyboardInterrupt:
        pass
    finally:
        rec.terminate()
        conn.close()
        srv.close()
        if hops:
            print(f'{hops} hops, mean filter time {1000 * t_proc / hops:.2f} ms/hop')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)

    def wiener_args(q):
        q.add_argument('-p', '--profile', required=True)
        q.add_argument('--over', type=float, default=1.5, help='noise over-estimation (more = stronger)')
        q.add_argument('--floor', type=float, default=0.0,
                       help='min gain per bin: 0.0 = best for SSL, 0.1 = smoother audio')
        q.add_argument('--dd', type=float, default=0.95, help='decision-directed smoothing 0..0.99')
        q.add_argument('--mask', type=float, default=0.0,
                       help='experimental: bins with gain below this set to 0 (0 = off)')

    q = sp.add_parser('profile', help='learn the known source (16-ch WAV, known source only)')
    q.add_argument('wav')
    q.add_argument('-o', '--out', default='wiener_profile.npz')
    q.add_argument('--keep-pct', type=float, default=25.0)

    q = sp.add_parser('apply', help='filter a 16-ch WAV offline')
    q.add_argument('wav')
    q.add_argument('-o', '--out', help='output base name (default: <input>_wiener)')
    wiener_args(q)

    q = sp.add_parser('check', help='compare SSL directions before/after')
    q.add_argument('before')
    q.add_argument('after')
    q.add_argument('-p', '--profile', required=True)
    q.add_argument('--tol', type=float, default=15.0, help='deg around the known direction')

    q = sp.add_parser('stream', help='real time: arecord -> Wiener -> ODAS socket')
    wiener_args(q)
    q.add_argument('--device', default='hw:2,0')
    q.add_argument('--port', type=int, default=9200)

    a = p.parse_args()
    {'profile': cmd_profile, 'apply': cmd_apply, 'check': cmd_check, 'stream': cmd_stream}[a.cmd](a)


if __name__ == '__main__':
    main()
