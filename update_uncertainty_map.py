import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize

def generate_uncertainty_map(x_min, x_max, y_min, y_max, z_min, z_max, interval, interval_z):
    """
    生成三维网格点，每个点保留原代码的 [x, y, z, 1.0] 结构。

    2026-10-03 性能优化：
    - 原版使用三重 Python for + append，场景变大后初始化很慢。
    - 这里改为 NumPy 向量化生成，网格范围、排列顺序和 dtype 均保持一致。
    """
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    z_start, z_end = min(z_min, z_max), max(z_min, z_max)

    xs = np.arange(x_start, x_end + interval, interval, dtype=np.float64)
    ys = np.arange(y_start, y_end + interval, interval, dtype=np.float64)
    zs = np.arange(z_start, z_end + interval_z, interval_z, dtype=np.float64)

    nx, ny, nz = len(xs), len(ys), len(zs)
    n = nx * ny * nz

    points = np.empty((n, 4), dtype=np.float64)
    points[:, 0] = np.repeat(xs, ny * nz)
    points[:, 1] = np.tile(np.repeat(ys, nz), nx)
    points[:, 2] = np.tile(zs, nx * ny)
    points[:, 3] = 1.0

    return points

def uncertainty_map_update(points, observer_pos, look_direction, step_x, fov=90, max_distance=100):
    """
    获取当前视野内的可见面并更新深度值。

    2026-10-03 性能优化：
    - 原版逐点 Python 循环。
    - 改为分块 NumPy 向量化，降低 Python 循环开销并控制临时内存。
    - 判断条件仍保持：
        distance <= max_distance
        distance > 0
        cos_angle > cos(fov/2)
    - 深度更新公式保持不变。
    """
    points = np.asarray(points)
    updated_points = points.copy()

    if len(points) == 0:
        return updated_points

    observer_pos = np.asarray(observer_pos, dtype=np.float64)
    look_direction = (
        np.asarray(look_direction, dtype=np.float64)
        * np.array([1.0, -1.0, 1.0], dtype=np.float64)
    )

    look_norm = np.linalg.norm(look_direction)
    if look_norm == 0:
        return updated_points

    look_direction_normalized = look_direction / look_norm
    cos_fov = np.cos(np.radians(fov / 2.0))

    # 控制峰值内存；对 Scene3/4 的大地图也能稳定运行。
    chunk_size = 500_000

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))

        xyz = points[start:end, :3]
        to_point = xyz - observer_pos
        distance_sq = np.einsum("ij,ij->i", to_point, to_point)

        valid = (distance_sq > 0.0) & (distance_sq <= float(max_distance) ** 2)
        if not np.any(valid):
            continue

        local_idx = np.flatnonzero(valid)
        vectors = to_point[local_idx]
        distances = np.sqrt(distance_sq[local_idx])

        cos_angle = (vectors @ look_direction_normalized) / distances
        visible_local = cos_angle > cos_fov

        if not np.any(visible_local):
            continue

        selected_local = local_idx[visible_local]
        selected_global = start + selected_local
        selected_dist = distances[visible_local]

        log_curr_dist = np.exp(-selected_dist * 0.15 / step_x)
        factor = 1.0 - log_curr_dist

        # 原代码的 face_depths = max(0, face_depths * factor)
        # 对当前 [x,y,z,depth] 数据结构与原实现等价；
        # np.maximum 同时兼容未来存在多列 depth 的情况。
        updated_points[selected_global, 3:] = np.maximum(
            0.0,
            updated_points[selected_global, 3:] * factor[:, None]
        )

    return updated_points


def cognitive_map_denoising(points, observer_pos, look_direction, step_x, fov=120, max_distance=2000):
    """
    获取当前视野内的认知地图点并清零。

    2026-10-03 性能优化：
    - 原版对整个 cognitive_map 逐点 Python 遍历。
    - Scene3 的 1m 网格可达到数百万点，因此单步会耗时几十秒。
    - 改为分块 NumPy 向量化，保持原版实际使用的 step_x * 1.9 距离条件。
    - max_distance 参数原代码未参与判定，这里为保持复现行为也不改变该逻辑。
    """
    points = np.asarray(points)
    updated_points = points.copy()

    if len(points) == 0:
        return updated_points

    observer_pos = np.asarray(observer_pos, dtype=np.float64)
    look_direction = (
        np.asarray(look_direction, dtype=np.float64)
        * np.array([1.0, -1.0, 1.0], dtype=np.float64)
    )

    look_norm = np.linalg.norm(look_direction)
    if look_norm == 0:
        return updated_points

    look_direction_normalized = look_direction / look_norm
    cos_fov = np.cos(np.radians(fov / 2.0))
    max_distance_effective = float(step_x) * 1.9
    max_distance_sq = max_distance_effective ** 2

    chunk_size = 500_000

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))

        xyz = points[start:end, :3]
        to_point = xyz - observer_pos
        distance_sq = np.einsum("ij,ij->i", to_point, to_point)

        valid = (distance_sq > 0.0) & (distance_sq <= max_distance_sq)
        if not np.any(valid):
            continue

        local_idx = np.flatnonzero(valid)
        vectors = to_point[local_idx]
        distances = np.sqrt(distance_sq[local_idx])

        cos_angle = (vectors @ look_direction_normalized) / distances
        visible_local = cos_angle > cos_fov

        if not np.any(visible_local):
            continue

        selected_global = start + local_idx[visible_local]
        updated_points[selected_global, 3] = 0.0
        updated_points[selected_global, 4] = 0.0

    return updated_points



def visualize_uncertainty_map(points, observer_pos, look_direction, step, if_figure_plot, filename_prefix='./output/uncertainty_map/uncertainty_map'):
    """可视化当前三维点的深度值，每个点表示为10x10x5的立方体"""
    look_direction = look_direction * [1, -1, 1]
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 创建颜色映射和归一化
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=1)

    # 立方体的尺寸
    cube_x, cube_y, cube_z = 5, 5, 5

    # 绘制每个点为立方体
    for point in points:
        x, y, z = point[:3]
        depths = point[3:]  # 获取六个面的深度值
        # 创建立方体的顶点
        x_vertices = [x, x, x + cube_x, x + cube_x, x, x, x + cube_x, x + cube_x]
        y_vertices = [y, y + cube_y, y, y + cube_y, y, y + cube_y, y, y + cube_y]
        z_vertices = [z, z, z, z, z + cube_z, z + cube_z, z + cube_z, z + cube_z]

        # 定义立方体的面
        vertices = np.column_stack((x_vertices, y_vertices, z_vertices))

        color = cmap(depths)

        # 使用深度值作为颜色强度
        for j in range(6):  # 6个面
            faces = [
                [vertices[0], vertices[1], vertices[3], vertices[2]],  # 底面
                [vertices[4], vertices[5], vertices[7], vertices[6]],  # 顶面
                [vertices[0], vertices[4], vertices[6], vertices[2]],  # 前面
                [vertices[1], vertices[5], vertices[7], vertices[3]],  # 后面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 左面
                [vertices[2], vertices[3], vertices[7], vertices[6]]  # 右面
            ]

            # 绘制每个面
            face_collection = Poly3DCollection([faces[j]], alpha=0.2, facecolor=color, edgecolor='k', linewidth=0.5)
            ax.add_collection3d(face_collection)

    # 绘制观察者位置和方向
    ax.scatter(*observer_pos, c='red', s=100, label='Observer')

    # 绘制观察方向
    ax.quiver(
        observer_pos[0], observer_pos[1], observer_pos[2],  # 起点坐标
        look_direction[0], look_direction[1], look_direction[2],  # 方向向量
        color='red',
        label='View Direction',
        length=20  # 添加长度参数
    )

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Grid Points Represented as Cubic Volumes with Depth Values')

    # 添加颜色条
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label='Depth Value', shrink=0.8, aspect=20)

    # 设置坐标轴范围
    x_min, x_max = points[:, 0].min() - 10, points[:, 0].max() + 20
    y_min, y_max = points[:, 1].min() - 10, points[:, 1].max() + 20
    z_min, z_max = points[:, 2].min() - 10, points[:, 2].max() + 20

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # ax.legend()
    plt.tight_layout()

    ax.grid(False)  # 关闭网格
    ax.set_axis_off()  # 隐藏坐标轴

    if if_figure_plot:
        plt.show()

    # 生成文件名
    filename = f"{filename_prefix}_step_{step}.png"  # 使用step命名文件

    # 保存图片
    plt.savefig(filename, bbox_inches='tight')  # 保存为图片，文件名为filename

    plt.close()

def is_target_visible(target_pos, observer_pos, look_direction, fov=120, max_distance=20):
    """判断目标位置是否在可见范围内"""
    look_direction = look_direction * [1, -1, 1]
    look_direction_normalized = look_direction / np.linalg.norm(look_direction)
    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance > max_distance:
        return False  # 超出最大距离

    if distance > 0:  # 计算可见性
        to_target_normalized = to_target / distance
        cos_angle = np.dot(to_target_normalized, look_direction_normalized)
        cos_fov = np.cos(np.radians(fov / 2))

        if cos_angle > cos_fov:  # 如果目标在视场范围内
            return True

    return False

def is_target_visible2(target_pos, observer_pos, look_direction, step_size, fov=120):
    """判断目标位置是否在可见范围内"""
    look_direction = look_direction * [1, -1, 1]
    look_direction_normalized = look_direction / np.linalg.norm(look_direction)
    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance > 2 * step_size:
        return False  # 超出最大距离

    if distance > 0:  # 计算可见性
        to_target_normalized = to_target / distance
        cos_angle = np.dot(to_target_normalized, look_direction_normalized)
        cos_fov = np.cos(np.radians(fov / 2))

        if cos_angle > cos_fov:  # 如果目标在视场范围内
            return True

    return False

def is_target_visible3(target_pos, observer_pos):
    """判断目标位置是否在可见范围内"""

    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance < 15:
        return True  # 超出最大距离
    else:
        return False



def main():
    # 初始化点数据
    points = generate_grid_points(6330, 6410, -4210, -4140, 1, 16, 10, 5)
    observer_pos = [6350, -4160, 6]
    look_direction = [1, 0, 0]
    total_depth = np.sum(points[:, 3:])
    print("总深度:", total_depth)
    # 获取可见面并更新深度值
    points = get_visible_faces_and_observe(points, observer_pos, look_direction, fov=45, max_distance=2000)

    visualize_depth_values_cubic(points, observer_pos, look_direction)

    total_depth = np.sum(points[:, 3:])
    print("更新后总深度:", total_depth)

    print(f"总点数: {len(points)}")

if __name__ == "__main__":
    main()
