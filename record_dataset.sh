#!/usr/bin/env bash
# record_dataset.sh - guided recording of S0..S4 (16 ch raw UMA-16) for wiener_analysis.py
# Usage: bash record_dataset.sh [seconds=30] [card=hw:2,0]
set -u
DUR=${1:-30}
DEV=${2:-hw:2,0}
RATE=44100
OUT=~/dataset
mkdir -p "$OUT"

# free the card: ODAS must be stopped
pkill -9 -f odas_core_node 2>/dev/null
pkill -9 -f odaslive 2>/dev/null
sleep 1
if ! arecord -l | grep -qi "uma\|minidsp"; then
  echo "[WARN] UMA-16 not found in 'arecord -l' - check USB and the card number:"
  arecord -l
fi

check() {  # quick per-take sanity check
python3 - "$1" <<'EOF'
import sys, numpy as np, soundfile as sf
x, sr = sf.read(sys.argv[1], dtype="float64", always_2d=True)
rms = 10*np.log10(np.mean(x**2, 0) + 1e-20)
clip = np.mean(np.abs(x) > 0.999)*100
dead = np.where(rms < np.median(rms) - 20)[0]
print(f"   {x.shape[1]} ch, {sr} Hz, {len(x)/sr:.1f} s | RMS {rms.min():.1f}..{rms.max():.1f} dBFS "
      f"| clip {clip:.3f}% | dead mics: {list(dead) or 'none'}")
if np.median(rms) < -80: print("   [WARN] almost silent - check the source / mic gain")
if clip > 0.01: print("   [WARN] clipping - move the source back or lower its volume")
EOF
}

take() {  # name, instruction
  local f="$OUT/$1.wav"
  echo
  echo "=================================================================="
  echo " $1  ($DUR s)"
  echo " $2"
  echo "=================================================================="
  read -rp " Set up the scene, then press ENTER to record (s = skip)... " a
  [ "$a" = "s" ] && return
  if ! arecord -D "$DEV" -c 16 -r $RATE -f S32_LE -d "$DUR" "$f"; then
    echo "[FAIL] arecord failed (device busy? run: sudo fuser -v /dev/snd/*)"; exit 1
  fi
  check "$f"
  read -rp " Keep this take? [Y/n] " k
  [ "$k" = "n" ] && take "$1" "$2"
}

echo "Record in a quiet room. Do NOT move the array or the known source between takes."
take S0_calib  "KNOWN source ON alone, at its fixed position (used to learn the profile)."
take S1_known  "KNOWN source ON alone, same position (test take, different from S0)."
take S2_target "Known source OFF. TARGET ON alone, at a position away from the known source."
take S3_both   "Known source ON + TARGET ON, target well away from the known direction."
take S4_cross  "Known source ON + TARGET moving slowly ACROSS the known source's direction."

echo
echo "Done. Files:"
ls -lh "$OUT"/*.wav
