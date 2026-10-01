# filter_before_ssl

Tests of a Wiener filter before ODAS sound source localization, and live known-source dropping (UMA-16 and UMA-8, ROS 2 Humble).

## Requirements
ROS 2 Humble + odas_ros, Python: `pip install "numpy<2" soundfile matplotlib pyyaml`

## Start ODAS
    bash odas_live.sh uma16.cfg        # or: bash odas_live.sh uma8.cfg
    bash odas_live.sh stop

## UMA-16 live (RViz)
    python3 known_track_drop_node.py --ros-args -p calib_seconds:=10.0 -p vertical:=true -p min_activity:=0.05 -p label_mode:=id
    rviz2 -d target_view.rviz

## UMA-8 live (RViz)
    python3 uma8_array_view_node.py --ros-args -p cfg:=$PWD/uma8.cfg
    python3 uma8_known_track_drop_node.py --ros-args -p calib_seconds:=10.0 -p min_activity:=0.05
    rviz2 -d uma8_view.rviz

## Offline filter tests
    bash record_dataset.sh 30                      # record S0..S4 (16 ch)
    python3 wiener_analysis.py --calib ... --cfg uma16.cfg   # step-by-step analysis
    python3 sss_test.py --cfg uma16.cfg --vertical --odas <path>/odaslive   # ODAS raw vs filtered, 4 SSS WAVs
    python3 listen.py                              # before/after listening page

## Files
| File | Role |
|---|---|
| odas_live.sh | clean start/stop of ODAS (any array) |
| known_track_drop_node.py | UMA-16: label tracks KNOWN/TARGET by ID, publish /sst_target |
| uma8_*.py, uma8_view.rviz | UMA-8 versions (topics under /uma8/) |
| target_view_node.py | earlier UMA-16 live view |
| wiener_analysis.py, sss_test.py, odas_ab_test.py, listen.py | offline Wiener-before-SSL tests |
| record_dataset.sh | guided 16-ch recording |
| uma16.cfg, uma8.cfg | ODAS configurations |

Notes: card numbers in the .cfg must match `arecord -l`. Recordings (*.wav, *.raw) are not in the repo.
