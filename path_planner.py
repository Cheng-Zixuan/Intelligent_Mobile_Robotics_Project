"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
import numpy as np
from heapq import heappush, heappop


def astar_3d(start, goal, env, grid_step=1.0):
    start = tuple(start)
    goal = tuple(goal)

    # 使用切比雪夫距离
    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

    open_set = []
    heappush(open_set, (heuristic(start, goal), start))

    came_from = {}
    g_cost = {start: 0}
    visited = set()

    # 预计算方向向量和代价
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # 对角线步长为√3，直线步长为1
                step_cost = np.linalg.norm([dx, dy, dz])
                directions.append((dx, dy, dz, step_cost))

    nodes_expanded = 0
    while open_set:
        _, current = heappop(open_set)
        nodes_expanded += 1

        if current in visited:
            continue
        visited.add(current)

        # 到达目标
        if heuristic(current, goal) < 0.5:
            path = reconstruct_path_optimized(came_from, current, env)
            # 简单而有效的路径平滑
            if len(path) > 3:
                path = simple_path_smooth(path, env)
            print(f"路径规划完成，扩展节点数: {nodes_expanded}")
            return path

        # 扩展邻居
        for dx, dy, dz, step_cost in directions:
            nxt = (current[0] + dx * grid_step,
                   current[1] + dy * grid_step,
                   current[2] + dz * grid_step)

            # 快速边界检查
            if (nxt[0] < 0 or nxt[0] > env.env_width or
                    nxt[1] < 0 or nxt[1] > env.env_length or
                    nxt[2] < 0 or nxt[2] > env.env_height):
                continue

            # 碰撞检查
            if env.is_collide(nxt):
                continue

            # 计算g代价（加入转向惩罚）
            new_g = g_cost[current] + step_cost

            # 转向惩罚：如果方向改变，增加微小代价
            if current in came_from:
                prev = came_from[current]
                prev_dir = np.array(current) - np.array(prev)
                curr_dir = np.array(nxt) - np.array(current)
                if np.linalg.norm(prev_dir / np.linalg.norm(prev_dir) -
                                  curr_dir / np.linalg.norm(curr_dir)) > 0.3:
                    new_g += 0.1  # 微小转向惩罚

            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                f = new_g + heuristic(nxt, goal)
                heappush(open_set, (f, nxt))
                came_from[nxt] = current

    raise RuntimeError("A* failed: No path found.")


def reconstruct_path_optimized(came_from, current, env):
    """尝试直线连接"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()

    # 尝试简化路径
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        # 检查是否能跳过中间点
        prev = simplified[-1]
        next_point = path[i + 1]
        current_point = path[i]

        # 如果prev到next_point是直线且无碰撞，跳过current
        if not is_collision_between(prev, next_point, env, samples=5):
            continue
        simplified.append(current_point)
    simplified.append(path[-1])

    return np.array(simplified)


def is_collision_between(p1, p2, env, samples=5):
    """检查两点间线段是否有碰撞"""
    for i in range(samples + 1):
        t = i / samples
        point = (p1[0] + (p2[0] - p1[0]) * t,
                 p1[1] + (p2[1] - p1[1]) * t,
                 p1[2] + (p2[2] - p1[2]) * t)
        if env.is_collide(point):
            return True
    return False


def simple_path_smooth(path, env, iterations=20):
    """梯度下降路径平滑"""
    if len(path) < 4:
        return path

    smoothed = path.copy()
    alpha = 0.1  # 吸引到原始路径
    beta = 0.3  # 平滑项权重

    for _ in range(iterations):
        new_path = smoothed.copy()
        for i in range(1, len(path) - 1):
            # 向原始点吸引
            move_to_original = (path[i] - smoothed[i]) * alpha
            # 向邻居中心吸引
            move_to_center = (smoothed[i - 1] + smoothed[i + 1] - 2 * smoothed[i]) * beta

            new_point = smoothed[i] + move_to_original + move_to_center

            # 确保不碰撞
            if not env.is_collide(tuple(new_point)):
                new_path[i] = new_point

        smoothed = new_path

    return smoothed