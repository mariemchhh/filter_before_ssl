#!/usr/bin/env python3
"""
uma8_array_view_node.py - Show the UMA-8 microphone array in RViz (UMA-16 setup untouched).

Reads the mic positions (mu = (x, y, z)) from the ODAS cfg and publishes /uma8/array_markers
(visualization_msgs/MarkerArray, latched) in frame "odas":
  - one sphere per mic + its index
  - the board outline (circle for a circular array, rectangle otherwise)
  - x / y / z axis labels, so you can match RViz directions with the az/el printed by the nodes

  source ~/ros2_ws/install/setup.bash
  python3 uma8_array_view_node.py --ros-args -p cfg:=$HOME/odas_ws/uma8.cfg
  rviz2 -d ~/Downloads/uma8_view.rviz
"""
import math
import os
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

SCALE = 3.0   # the array is drawn 3x larger so the mics are visible next to the 0.5 m arrows


def parse_mics(path):
    txt = open(os.path.expanduser(path)).read()
    m = txt.find("mics")
    txt = txt[m:] if m >= 0 else txt
    num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    pts = re.findall(r"mu\s*=\s*\(\s*" + num + r"\s*,\s*" + num + r"\s*,\s*" + num + r"\s*\)", txt)
    return [(float(a), float(b), float(c)) for a, b, c in pts]


class Uma8ArrayView(Node):
    def __init__(self):
        super().__init__("uma8_array_view")
        self.declare_parameter("cfg", os.path.expanduser("~/odas_ws/uma8.cfg"))
        self.declare_parameter("frame", "odas")
        self.frame = self.get_parameter("frame").value
        cfg = self.get_parameter("cfg").value
        self.mics = parse_mics(cfg)
        if not self.mics:
            raise SystemExit(f"no 'mu = (x, y, z)' mic positions found in {cfg}")
        self.get_logger().info(f"{len(self.mics)} mics read from {cfg}")
        for i, (x, y, z) in enumerate(self.mics):
            self.get_logger().info(f"  mic {i}: x {x*1000:6.1f} mm  y {y*1000:6.1f} mm  z {z*1000:6.1f} mm")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(MarkerArray, "/uma8/array_markers", qos)
        self.publish()
        self.create_timer(2.0, self.publish)   # republish for late RViz start

    def base(self, mid, mtype, ns):
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def publish(self):
        ma = MarkerArray()
        P = [(x * SCALE, y * SCALE, z * SCALE) for x, y, z in self.mics]
        cx = sum(p[0] for p in P) / len(P); cy = sum(p[1] for p in P) / len(P); cz = sum(p[2] for p in P) / len(P)

        for i, (x, y, z) in enumerate(P):
            s = self.base(i, Marker.SPHERE, "mics")
            s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, z
            s.scale.x = s.scale.y = s.scale.z = 0.025
            s.color.r, s.color.g, s.color.b, s.color.a = 0.95, 0.85, 0.2, 1.0
            ma.markers.append(s)
            t = self.base(100 + i, Marker.TEXT_VIEW_FACING, "mic_ids")
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z + 0.035
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = str(i)
            ma.markers.append(t)

        # board outline: circle if all outer mics are at ~the same radius, else bounding rectangle
        r = [math.dist((x, y, z), (cx, cy, cz)) for x, y, z in P]
        outer = [v for v in r if v > 1e-6]
        line = self.base(200, Marker.LINE_STRIP, "board")
        line.scale.x = 0.006
        line.color.r, line.color.g, line.color.b, line.color.a = 0.6, 0.7, 0.8, 0.9
        span = [max(p[k] for p in P) - min(p[k] for p in P) for k in range(3)]
        flat = span.index(min(span))                     # axis normal to the board
        a1, a2 = [k for k in range(3) if k != flat]
        c = [cx, cy, cz]
        if outer and (max(outer) - min(outer)) < 0.15 * max(outer):     # circular (UMA-8)
            R = max(outer) * 1.25
            for k in range(65):
                th = 2 * math.pi * k / 64
                q = list(c); q[a1] += R * math.cos(th); q[a2] += R * math.sin(th)
                line.points.append(Point(x=q[0], y=q[1], z=q[2]))
        else:                                                           # rectangular (UMA-16)
            lo = [min(p[k] for p in P) for k in range(3)]; hi = [max(p[k] for p in P) for k in range(3)]
            pad = 0.15 * max(span)
            for u, v in ((lo[a1] - pad, lo[a2] - pad), (hi[a1] + pad, lo[a2] - pad),
                         (hi[a1] + pad, hi[a2] + pad), (lo[a1] - pad, hi[a2] + pad), (lo[a1] - pad, lo[a2] - pad)):
                q = list(c); q[a1], q[a2] = u, v
                line.points.append(Point(x=q[0], y=q[1], z=q[2]))
        ma.markers.append(line)

        for k, (name, col) in enumerate((("x", (1.0, 0.3, 0.3)), ("y", (0.3, 1.0, 0.3)), ("z", (0.4, 0.6, 1.0)))):
            t = self.base(300 + k, Marker.TEXT_VIEW_FACING, "axes")
            setattr(t.pose.position, name, 0.2)
            t.scale.z = 0.05
            t.color.r, t.color.g, t.color.b = (float(c) for c in col); t.color.a = 1.0
            t.text = name
            ma.markers.append(t)

        title = self.base(400, Marker.TEXT_VIEW_FACING, "title")
        title.pose.position.z = -0.12
        title.scale.z = 0.04
        title.color.r = title.color.g = title.color.b = 0.8; title.color.a = 1.0
        title.text = f"{len(self.mics)}-mic array (drawn x{SCALE:.0f})"
        ma.markers.append(title)
        self.pub.publish(ma)


def main():
    rclpy.init()
    n = Uma8ArrayView()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
