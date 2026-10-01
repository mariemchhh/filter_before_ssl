#!/usr/bin/env python3
"""
uma8_known_track_drop_node.py - UMA-8 version: drop the known source's SST track by its ID, follow the TARGET.
Topics are under /uma8/ so the UMA-16 setup is untouched.

Subscribes /sst (odas_ros_msgs/OdasSstArrayStamped), labels every active track:
  KNOWN  (red)   : the known source
  TARGET (green) : anything else
label_mode:=id (default)  -> each track ID is labeled ONCE, when it first appears
                             (inside the cone = KNOWN, outside = TARGET) and keeps that label,
                             so a target crossing the known direction stays TARGET (no filter needed)
label_mode:=direction     -> every frame by angle (target inside the cone shows as KNOWN)
Also publishes /uma8/sst_target: same message with the KNOWN tracks dropped (id=0, activity=0),
slot order kept so it still matches the SSS channels.
Publishes:
  /uma8/target_markers  visualization_msgs/MarkerArray  -> RViz (arrows + "TARGET az/el" text)
  /uma8/target_status   std_msgs/String (JSON: detected, az, el, id, activity)
Prints a status line twice per second and logs every frame to ~/dataset/uma8_known_track_drop_<time>.csv

Known direction:
  - learned at start: calib_seconds (default 10 s) with ONLY the known source on, or
  - given: -p known_az:=16.0 -p known_el:=-3.0  (degrees, same convention as below)

Run (ODAS live already publishing /sst):
  source ~/ros2_ws/install/setup.bash
  python3 uma8_known_track_drop_node.py --ros-args -p calib_seconds:=10.0 -p vertical:=true -p label_mode:=id
RViz: Fixed Frame = odas, Add -> By topic -> /uma8/target_markers (MarkerArray)
"""
import copy
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from odas_ros_msgs.msg import OdasSstArrayStamped
except ImportError as e:
    raise SystemExit(f"odas_ros_msgs not found ({e}). Run: source ~/ros2_ws/install/setup.bash")


def unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1e-12
    return [c / n for c in v]


def angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(unit(a), unit(b)))))))


class Uma8KnownTrackDrop(Node):
    def __init__(self):
        super().__init__("uma8_known_track_drop")
        self.declare_parameter("calib_seconds", 10.0)
        self.declare_parameter("known_az", float("nan"))
        self.declare_parameter("known_el", float("nan"))
        self.declare_parameter("cone_deg", 30.0)
        self.declare_parameter("min_activity", 0.3)
        self.declare_parameter("vertical", False)
        self.declare_parameter("sst_topic", "/sst")
        self.declare_parameter("frame", "odas")
        self.declare_parameter("label_mode", "id")
        g = lambda n: self.get_parameter(n).value
        self.cone, self.min_act, self.vertical = g("cone_deg"), g("min_activity"), g("vertical")
        self.frame = g("frame")
        self.mode = g("label_mode")
        self.id_label = {}                 # track id -> "KNOWN" / "TARGET" (id mode)
        self.calib_s = g("calib_seconds")
        az, el = g("known_az"), g("known_el")
        self.known = None if math.isnan(az) or math.isnan(el) else self.dir_from(az, el)
        self.calib_acc, self.t0 = [], None

        self.pub_m = self.create_publisher(MarkerArray, "/uma8/target_markers", 10)
        self.pub_s = self.create_publisher(String, "/uma8/target_status", 10)
        self.pub_t = self.create_publisher(OdasSstArrayStamped, "/uma8/sst_target", 10)
        self.create_subscription(OdasSstArrayStamped, g("sst_topic"), self.on_sst, 10)
        self.create_timer(0.5, self.print_status)

        os.makedirs(os.path.expanduser("~/dataset"), exist_ok=True)
        self.csv_path = os.path.expanduser(f"~/dataset/uma8_known_track_drop_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.csv = open(self.csv_path, "w")
        self.csv.write("t,id,x,y,z,az,el,activity,label,angle_to_known\n")
        self.last, self.n_frames, self.n_target = None, 0, 0
        self.n_msgs, self.n_tracks_calib, self.t_start = 0, 0, time.time()
        self.t_last_msg, self.prev_n = None, 0
        msg = "known direction given" if self.known else f"CALIBRATION {self.calib_s:.0f} s: only the KNOWN source on!"
        self.get_logger().info(msg)

    # ---------- geometry (vertical board: x horizontal, y vertical, z depth)
    def az_el(self, v):
        x, y, z = unit(v)
        if self.vertical:
            return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, math.hypot(x, z)))
        return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))

    def dir_from(self, az, el):
        a, e = math.radians(az), math.radians(el)
        if self.vertical:
            return [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]
        return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]

    # ---------- main callback
    def on_sst(self, msg):
        now = time.time()
        self.n_msgs += 1
        self.t_last_msg = now
        srcs = [s for s in msg.sources if s.id != 0 and s.activity >= self.min_act]

        if self.known is None:                                   # calibration phase
            self.t0 = self.t0 or now
            if srcs:
                self.n_tracks_calib += 1
                for s_ in srcs:
                    self.id_label[s_.id] = "KNOWN"
                best = max(srcs, key=lambda s: s.activity)
                self.calib_acc.append(unit([best.x, best.y, best.z]))
            if now - self.t0 >= self.calib_s:
                if not self.calib_acc:
                    self.get_logger().warn("no track during calibration - is the known source on? retrying")
                    self.t0 = now
                    return
                m = unit([sum(v[i] for v in self.calib_acc) for i in range(3)])
                self.known = m
                az, el = self.az_el(m)
                self.get_logger().info(f"KNOWN direction learned: az {az:.1f} deg, el {el:.1f} deg "
                                       f"-> now turn the TARGET on")
            return

        ma = MarkerArray()
        d = Marker(); d.action = Marker.DELETEALL; ma.markers.append(d)
        ma.markers.append(self.arrow(9999, self.known, (1.0, 0.2, 0.2, 0.25), 0.6, "known_ref"))
        target = None
        for i, s in enumerate(srcs):
            v = unit([s.x, s.y, s.z])
            a2k = angle(v, self.known)
            if self.mode == "direction":
                label = "KNOWN" if a2k < self.cone else "TARGET"
            else:
                if s.id not in self.id_label:     # new track: decide once, then keep
                    self.id_label[s.id] = "KNOWN" if a2k < self.cone else "TARGET"
                    az0, el0 = self.az_el(v)
                    self.get_logger().info(f"new track id {s.id} at az {az0:.0f} el {el0:.0f} "
                                           f"-> {self.id_label[s.id]}")
                label = self.id_label[s.id]
            az, el = self.az_el(v)
            col = (1.0, 0.1, 0.1, 1.0) if label == "KNOWN" else (0.1, 1.0, 0.2, 1.0)
            ma.markers.append(self.arrow(2 * i, v, col, 0.5, label))
            ma.markers.append(self.text(2 * i + 1, v, f"{label} id{s.id}\naz {az:.0f} el {el:.0f}", col))
            self.csv.write(f"{now:.3f},{s.id},{s.x:.3f},{s.y:.3f},{s.z:.3f},{az:.1f},{el:.1f},"
                           f"{s.activity:.2f},{label},{a2k:.1f}\n")
            if label == "TARGET" and (target is None or s.activity > target[0].activity):
                target = (s, az, el)
        self.pub_m.publish(ma)

        out = copy.deepcopy(msg)                   # drop KNOWN tracks, keep slot order
        for src in out.sources:
            if src.id != 0 and self.id_label.get(src.id) == "KNOWN" and self.mode == "id" or \
               (self.mode == "direction" and src.id != 0 and
                angle(unit([src.x, src.y, src.z]), self.known) < self.cone):
                src.id, src.activity, src.x, src.y, src.z = 0, 0.0, 0.0, 0.0, 0.0
        self.pub_t.publish(out)

        st = {"detected": target is not None}
        if target:
            s, az, el = target
            st.update(id=int(s.id), az=round(az, 1), el=round(el, 1), activity=round(float(s.activity), 2))
            self.n_target += 1
        self.n_frames += 1
        self.last = (st, len(srcs))
        self.pub_s.publish(String(data=json.dumps(st)))

    def arrow(self, mid, v, col, length, ns):
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, Marker.ARROW, Marker.ADD
        from geometry_msgs.msg import Point
        m.points = [Point(), Point(x=v[0] * length, y=v[1] * length, z=v[2] * length)]
        m.scale.x, m.scale.y, m.scale.z = 0.02, 0.04, 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = col
        return m

    def text(self, mid, v, txt, col):
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "labels", mid, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (c * 0.6 for c in v)
        m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = col
        m.text = txt
        return m

    def print_status(self):
        if self.n_msgs == 0:
            print(f"\rwaiting for /sst ... no message after {time.time()-self.t_start:4.0f} s "
                  f"(is odas_core_node running?)   ", end="", flush=True)
            return
        if self.known is None:
            left = self.calib_s - (time.time() - (self.t0 or time.time()))
            print(f"\rCALIBRATION: /sst msgs {self.n_msgs}, frames with a track {self.n_tracks_calib}, "
                  f"{max(0, left):4.1f} s left   ", end="", flush=True)
            return
        if self.last is None:
            return
        rate = (self.n_msgs - self.prev_n) * 2.0
        self.prev_n = self.n_msgs
        if self.t_last_msg and time.time() - self.t_last_msg > 1.0:
            print(f"\r/sst STOPPED: no message for {time.time()-self.t_last_msg:4.0f} s "
                  f"- check the odas_core_node terminal              ", end="", flush=True)
            return
        st, n = self.last
        share = 100.0 * self.n_target / max(1, self.n_frames)
        if st["detected"]:
            print(f"\r[{rate:3.0f} Hz] TARGET  id {st['id']:>3}  az {st['az']:7.1f}  el {st['el']:6.1f}  act {st['activity']:.2f}"
                  f" | tracks {n} | target seen {share:5.1f}% of frames   ", end="", flush=True)
        else:
            kid = ",".join(str(k) for k, v in self.id_label.items() if v == "KNOWN") or "-"
            print(f"\r[{rate:3.0f} Hz] no target  (known ids {kid}) | tracks {n} | target seen {share:5.1f}% of frames"
                  f"                     ", end="", flush=True)

    def destroy_node(self):
        self.csv.close()
        print(f"\nlog -> {self.csv_path}")
        super().destroy_node()


def main():
    rclpy.init()
    n = Uma8KnownTrackDrop()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
