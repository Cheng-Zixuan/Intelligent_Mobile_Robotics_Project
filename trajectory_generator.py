"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def generate_trajectory(path, total_time=12.0):

    path = np.array(path)
    if len(path) < 2:
        return {"t": [0], "x": [path[0, 0]], "y": [path[0, 1]], "z": [path[0, 2]], "path": path}

    # 计算路径段长度
    segment_lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    total_length = np.sum(segment_lengths)

    # 自适应总时间（基于长度和最大速度估计）
    if total_time is None:
        avg_speed = 1.5  # 假设平均速度
        total_time = total_length / avg_speed

    # 非均匀时间分配：按段长比例分配时间
    time_nodes = np.zeros(len(path))
    time_nodes[1:] = np.cumsum(segment_lengths / total_length * total_time)

    # 三次样条插值，使用自然边界条件
    cs_x = CubicSpline(time_nodes, path[:, 0], bc_type='natural')
    cs_y = CubicSpline(time_nodes, path[:, 1], bc_type='natural')
    cs_z = CubicSpline(time_nodes, path[:, 2], bc_type='natural')

    # 生成密集轨迹点
    t_dense = np.linspace(0, total_time, min(500, int(total_time * 50)))

    # 计算速度（一阶导数）
    x_dense = cs_x(t_dense)
    y_dense = cs_y(t_dense)
    z_dense = cs_z(t_dense)

    # 检查速度约束
    vx = cs_x(t_dense, 1)
    vy = cs_y(t_dense, 1)
    vz = cs_z(t_dense, 1)
    speeds = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    max_speed = np.max(speeds)
    if max_speed > 3.0:  # 如果超过最大速度限制，重新调整时间
        print(f"最大速度 {max_speed:.2f}m/s 超过限制，自动调整时间")
        total_time *= max_speed / 2.5  # 延长总时间
        time_nodes *= total_time / time_nodes[-1]
        cs_x = CubicSpline(time_nodes, path[:, 0], bc_type='natural')
        cs_y = CubicSpline(time_nodes, path[:, 1], bc_type='natural')
        cs_z = CubicSpline(time_nodes, path[:, 2], bc_type='natural')
        t_dense = np.linspace(0, total_time, min(500, int(total_time * 50)))
        x_dense, y_dense, z_dense = cs_x(t_dense), cs_y(t_dense), cs_z(t_dense)

    return {
        "t": t_dense,
        "x": x_dense,
        "y": y_dense,
        "z": z_dense,
        "path": path,
        "t_nodes": time_nodes,
        "total_length": total_length,
         }

def plot_trajectory(traj):
    t = traj["t"]
    path = traj["path"]
    t_nodes = traj["t_nodes"]

    plt.figure(figsize=(12, 8))

    # X plot
    plt.subplot(3, 1, 1)
    plt.plot(t, traj["x"], label="x(t)")
    plt.scatter(t_nodes, path[:, 0], color="red", label="path points")
    plt.ylabel("x/(m)")
    plt.legend()

    # Y plot
    plt.subplot(3, 1, 2)
    plt.plot(t, traj["y"], label="y(t)")
    plt.scatter(t_nodes, path[:, 1], color="red")
    plt.ylabel("y/(m)")

    # Z plot
    plt.subplot(3, 1, 3)
    plt.plot(t, traj["z"], label="z(t)")
    plt.scatter(t_nodes, path[:, 2], color="red")
    plt.ylabel("z/(m)")
    plt.xlabel("time (s)")

    plt.tight_layout()
    plt.show()