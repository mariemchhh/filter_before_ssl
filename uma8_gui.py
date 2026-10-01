#!/usr/bin/env python3
# uma8_gui.py - live graphical DOA for miniDSP UMA-8.
#   RAW firmware (8ch): SRP-PHAT spatial spectrum from the 7 MEMS mics.
#   DSP firmware (2ch): on-chip XVF3000 DOA over USB (experimental) + levels.
import math, queue, sys, struct
import numpy as np, sounddevice as sd
import matplotlib.pyplot as plt

FS, NFFT, BLOCK = 48000, 512, 4096
FREQ = [500, 4000]                 # below ~4 kHz spatial aliasing
R = 0.043                          # ring radius (m)
MICS = [0, 1, 2, 3, 4, 5, 6]       # 0-based: center, ring 0..300° CCW from +x
ALGO = "SRP"                       # or "MUSIC", "NormMUSIC", "TOPS"
RMS_GATE = 1e-4

dev = next((i for i, d in enumerate(sd.query_devices())
            if ("uma" in d["name"].lower() or "minidsp" in d["name"].lower())
            and d["max_input_channels"] > 0), None)
if dev is None:
    sys.exit("UMA-8 not found (PulseAudio or odaslive may be holding it)")
info = sd.query_devices(dev); CH = info["max_input_channels"]; RAW = CH >= 7
print(f"Device: {info['name']}  channels={CH}  mode={'RAW' if RAW else 'DSP'}")

q = queue.Queue(maxsize=4)
def cb(indata, frames, t, status):
    try: q.put_nowait(indata.copy())
    except queue.Full: pass

usbdev = None
if RAW:
    import pyroomacoustics as pra
    geom = np.array([[0.0, 0.0]] + [[R*math.cos(math.radians(60*k)), R*math.sin(math.radians(60*k))]
                                     for k in range(6)]).T
    doa = pra.doa.algorithms[ALGO](geom, FS, NFFT, c=343.0, num_src=1, n_grid=360)
else:
    try:
        import usb.core
        usbdev = usb.core.find(idVendor=0x2752, idProduct=0x001c)
    except ImportError:
        print("pyusb not installed -> no on-chip DOA")

def xvf_doa():
    r = usbdev.ctrl_transfer(0xC0, 0, 0xC0, 21, 8, 500)   # vendor IN, param 21 = DOAANGLE
    return struct.unpack("ii", bytes(r))[0]

plt.ion()
fig = plt.figure(figsize=(11, 5.5))
ax = fig.add_subplot(1, 2, 1, projection="polar")
axl = fig.add_subplot(1, 2, 2)
ax.set_ylim(0, 1.05); ax.set_yticklabels([])
spec, = ax.plot([], [], lw=2)
arrow, = ax.plot([0, 0], [0, 0], color="red", lw=4)
title = ax.set_title("waiting for audio...")
labels = [f"ch{i+1}" for i in range(CH)]
bars = axl.bar(labels, np.zeros(CH), bottom=-100)
axl.set_ylim(-100, 0); axl.set_ylabel("dBFS"); axl.set_title("Channel levels")
fig.tight_layout()

with sd.InputStream(device=dev, channels=CH, samplerate=FS, blocksize=BLOCK,
                    dtype="float32", callback=cb):
    while plt.fignum_exists(fig.number):
        try:
            x = q.get(timeout=0.5)
        except queue.Empty:
            plt.pause(0.01); continue

        db = 20*np.log10(np.maximum(np.sqrt((x**2).mean(0)), 1e-9))
        for b, v in zip(bars, db):
            b.set_height(max(v, -100) + 100)

        if RAW:
            xs = x[:, MICS]; rms = float(np.sqrt((xs**2).mean()))
            if rms >= RMS_GATE:
                S = pra.transform.stft.analysis(xs, NFFT, NFFT // 2)
                doa.locate_sources(np.transpose(S, (2, 1, 0)), freq_range=FREQ)
                v = np.asarray(doa.grid.values, float)
                v = (v - v.min()) / (np.ptp(v) + 1e-12)
                th = np.asarray(doa.grid.azimuth, float)
                spec.set_data(np.append(th, th[0]), np.append(v, v[0]))
                az = float(doa.azimuth_recon[0])
                arrow.set_data([az, az], [0, 1])
                title.set_text(f"SRP-PHAT  az = {math.degrees(az) % 360:.0f}°")
            else:
                title.set_text(f"quiet (rms={rms:.5f})")
        elif usbdev is not None:
            try:
                a = math.radians(xvf_doa())
                arrow.set_data([a, a], [0, 1])
                title.set_text(f"On-chip XVF DOA = {math.degrees(a):.0f}°")
            except Exception as e:
                print("on-chip DOA read failed:", e); usbdev = None
        else:
            title.set_text("DSP firmware: no DOA - flash RAW firmware")

        plt.pause(0.01)
