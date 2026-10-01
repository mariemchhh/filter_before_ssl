#!/usr/bin/env python3
# Live polar viewer for ODAS + debug stats. Listens on 9001 (potentials) and 9000 (tracked).
import socket, threading, json, math, collections, time
import matplotlib.pyplot as plt

POT_PORT, TRK_PORT = 9001, 9000
E_MIN = 0.0             # show everything while debugging
HIST = 300

lock = threading.Lock()
pots = collections.deque(maxlen=HIST)
trk = {}
stats = {"pot_frames": 0, "trk_frames": 0, "E_max": 0.0, "E_sum": 0.0, "E_n": 0, "trk_ids": set()}

def az_el(s):
    x, y, z = s.get("x", 0.0), s.get("y", 0.0), s.get("z", 0.0)
    n = math.sqrt(x*x + y*y + z*z) or 1.0
    return math.atan2(y, x), math.degrees(math.asin(max(-1, min(1, z / n))))

def handle(obj, kind):
    src = obj.get("src", [])
    with lock:
        if kind == "pot":
            stats["pot_frames"] += 1
            for s in src:
                E = s.get("E", 0.0)
                stats["E_max"] = max(stats["E_max"], E); stats["E_sum"] += E; stats["E_n"] += 1
                if E >= E_MIN:
                    a, e = az_el(s); pots.append((a, e, E))
        else:
            stats["trk_frames"] += 1
            trk.clear()
            for s in src:
                if s.get("id", 0) != 0:
                    trk[s["id"]] = az_el(s); stats["trk_ids"].add(s["id"])

def serve(port, kind):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(1)
    dec = json.JSONDecoder()
    while True:
        c, addr = srv.accept(); print(f"[{port}] ODAS connected from {addr}")
        buf, first = "", True
        while True:
            d = c.recv(65536)
            if not d: break
            buf += d.decode(errors="ignore")
            if first:
                print(f"[{port}] first bytes: {buf[:300]!r}"); first = False
            while True:
                buf = buf.lstrip()
                if not buf: break
                try: obj, i = dec.raw_decode(buf)
                except json.JSONDecodeError: break
                buf = buf[i:]; handle(obj, kind)
        print(f"[{port}] ODAS disconnected")

def reporter():
    while True:
        time.sleep(1.0)
        with lock:
            n = stats["E_n"] or 1
            print(f"pot fr/s={stats['pot_frames']:4d}  trk fr/s={stats['trk_frames']:4d}  "
                  f"E_max={stats['E_max']:.3f}  E_mean={stats['E_sum']/n:.3f}  "
                  f"tracked ids={sorted(stats['trk_ids'])}")
            stats.update(pot_frames=0, trk_frames=0, E_max=0.0, E_sum=0.0, E_n=0, trk_ids=set())

for p, k in ((POT_PORT, "pot"), (TRK_PORT, "trk")):
    threading.Thread(target=serve, args=(p, k), daemon=True).start()
threading.Thread(target=reporter, daemon=True).start()

plt.ion()
fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(projection="polar")
while plt.fignum_exists(fig.number):
    with lock:
        P = list(pots); T = dict(trk)
    ax.clear(); ax.set_ylim(0, 90); ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["60°", "30°", "0° elev"]); ax.set_title("ODAS DOA (0° = +x, CCW)")
    if P:
        ax.scatter([p[0] for p in P], [90 - p[1] for p in P],
                   c=[p[2] for p in P], cmap="viridis", vmin=0, vmax=1, s=12, alpha=0.6)
    for i, (a, e) in T.items():
        ax.scatter([a], [90 - e], s=250, c="red", edgecolors="k")
        ax.annotate(f"id {i}\n{math.degrees(a):.0f}°", (a, 90 - e), color="red",
                    xytext=(8, 8), textcoords="offset points")
    plt.pause(0.05)
