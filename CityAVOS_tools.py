# ============================================================

# CityAVOS_tools.py

# 修改记录：

# 2026-09-28

# - 保留原作者代码，不删除；被替换的原代码已在对应位置完整注释并标注“原作者代码”。

# - 修正 get_action_from_llm() 中两张图片的输入顺序。

# - 增强 LLM 动作返回值解析，兼容 "Turn Left." 等带标点/大小写差异的结果。

#

# 2026-10-02 修改

# - 新增 predict_next_position() / is_position_in_scene() / get_valid_action_names()。

#   作用：限制最终动作不飞出当前 Scene。

# - get_action_from_llm() 新增 valid_actions。

#   作用：把当前合法动作传给 LLM。

# - 新增 Qwen 单次请求测速。

#   作用：区分“单次 API 很慢”和“多次重试累计很慢”，不改变决策逻辑。

# 2026-10-03 修改

# - get_action_from_llm() 新增 action_history / position_history。

#   作用：给本地 VLM 提供最近动作/位置，避免无记忆导航。

# - 新增 build_navigation_history_hint()。

#   作用：检测 Go Left <-> Go Right / Turn Left <-> Turn Right 往返，

#   以及“走两步回到原位”的位置循环。

# - 循环检测仅作为提示，不再从 Prompt 合法动作集中硬删除动作；

#   main.py 的 Scene 硬边界保护仍然保留。

# ============================================================



# ============================================================



import numpy as np

import math

import time

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN

from llm_agent import chat_with_llm_images

from update_uncertainty_map import uncertainty_map_update



# 定义动作集合

action_set = ["Go Up", "Go Down", "Turn Left", "Turn Right", "Go Forward", "Go Left", "Go Right", "Stop"]





def rad_to_deg(rad):

    angle_rad = np.deg2rad(rad)

    # 计算 x 和 y 分量

    x = np.cos(angle_rad)

    y = np.sin(angle_rad)

    # 创建方向向量，z 分量为 0

    direction = np.array([x, y, 0])

    return direction



# ============================================================

# 2026-10-02 新增：最终动作边界保护辅助函数

#

# 作用：

# 1. predict_next_position() 只预测位置，不实际移动无人机。

# 2. is_position_in_scene() 统一检查 x/y/z 是否仍位于当前 Scene。

# 3. get_valid_action_names() 给 LLM 提供“当前允许动作”列表。

#

# 注意：

# - 坐标系统继续沿用当前项目的 map 坐标：drone.pos * [1, -1, -1]。

# - Turn Left / Turn Right 不改变位置，因此始终不会因位置越界被拒绝。

# - Stop 不参与默认移动动作列表；模型明确返回 Stop 时仍允许通过。

# ============================================================

def predict_next_position(drone, action_label, scene):

    curr_pos = np.asarray(

        drone.pos * [1, -1, -1],

        dtype=float

    ).copy()



    yaw = float(drone.ori[2])

    step_x = float(scene["step_x"])

    step_z = float(scene["step_z"])



    next_pos = curr_pos.copy()



    # 0: Go Up

    if action_label == 0:

        next_pos[2] += step_z



    # 1: Go Down

    elif action_label == 1:

        next_pos[2] -= step_z



    # 2 / 3: Turn Left / Turn Right，只改变朝向，不改变位置

    elif action_label in (2, 3):

        pass



    # 4: Go Forward

    elif action_label == 4:

        next_pos[0] += step_x * np.cos(math.radians(yaw))

        next_pos[1] -= step_x * np.sin(math.radians(yaw))



    # 5: Go Left

    elif action_label == 5:

        next_pos[0] += step_x * np.cos(math.radians(yaw - 90))

        next_pos[1] -= step_x * np.sin(math.radians(yaw - 90))



    # 6: Go Right

    elif action_label == 6:

        next_pos[0] += step_x * np.cos(math.radians(yaw + 90))

        next_pos[1] -= step_x * np.sin(math.radians(yaw + 90))



    # 7: Stop，不改变位置

    elif action_label == 7:

        pass



    else:

        raise ValueError(f"Unknown action_label: {action_label}")



    # 消除 cos(90°) 等带来的 1e-15 级浮点残差，方便边界比较和日志查看

    return np.round(next_pos, 8)





def is_position_in_scene(pos, scene, eps=1e-6):

    pos = np.asarray(pos, dtype=float)



    return (

        scene["x_min"] - eps <= pos[0] <= scene["x_max"] + eps

        and scene["y_min"] - eps <= pos[1] <= scene["y_max"] + eps

        and scene["z_min"] - eps <= pos[2] <= scene["z_max"] + eps

    )





def get_valid_action_names(drone, scene, include_stop=False):

    valid_actions = []



    # 默认检查 0~6；Stop 由参数决定是否加入。

    max_action_index = 8 if include_stop else 7



    for action_label in range(max_action_index):

        next_pos = predict_next_position(

            drone,

            action_label,

            scene

        )



        if is_position_in_scene(next_pos, scene):

            valid_actions.append(action_set[action_label])



    return valid_actions





# 作者原代码

# def action_value_choose(drone, points, scene, total_uncertainty):



# 修改后

def action_value_choose(drone, points, scene, total_uncertainty,theta_T=0.1):

    points_now = points.copy()

    uncertainty_now = np.sum(points[:, 3:])

    total_depth = [0, 0, 0, 0, 0, 0, 0]

    [x_min, x_max, y_min, y_max, z_min, z_max, step_x, step_z] = [scene["x_min"], scene["x_max"], scene["y_min"],

                                                                  scene["y_max"], scene["z_min"], scene["z_max"],

                                                                  scene["step_x"], scene["step_z"]]



    # 检查位置是否超出边界的函数

    def is_out_of_bounds(pos):

        return (pos[0] < x_min or pos[0] > x_max or

                pos[1] < y_min or pos[1] > y_max or

                pos[2] < z_min or pos[2] > z_max)



    # 位置变换和观测

    positions = [

        drone.pos * [1, -1, -1] + [0, 0, step_z],  # 上

        drone.pos * [1, -1, -1] - [0, 0, step_z],  # 下

        drone.pos * [1, -1, -1],

        drone.pos * [1, -1, -1],

        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2])),

                                   -step_x * np.sin(math.radians(drone.ori[2])), 0],

        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2] - 90)),

                                   -step_x * np.sin(math.radians(drone.ori[2] - 90)), 0],

        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2] + 90)),

                                   -step_x * np.sin(math.radians(drone.ori[2] + 90)), 0]

    ]



    look_directions = [

        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 上

        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 下

        np.round(rad_to_deg(drone.ori[2] - 90)).astype(int),  # 左

        np.round(rad_to_deg(drone.ori[2] + 90)).astype(int),  # 右

        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 前进

        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 前进

        np.round(rad_to_deg(drone.ori[2])).astype(int)  # 前进

    ]



    # 遍历并计算深度

    for i in range(7):

        # 检查是否超出边界

        if is_out_of_bounds(positions[i]):

            total_depth[i] = 10000

        else:

            # 注意：这里调用了 update_uncertainty_map 中的函数

            points_now = uncertainty_map_update(points, positions[i], look_directions[i], step_x, fov=60,

                                                       max_distance=1000)

            total_depth[i] = np.sum(points_now[:, 3:])



    index = np.argmin(total_depth)



    temp = (uncertainty_now - total_depth[index]) / total_uncertainty

    print(temp)



# -------------------------------------修改线--------------



#    if temp > 0.1:

#        return action_set[np.argmin(total_depth)]

#   else:

#       return None



# ============================================================

# 2026-09-29 修改

# 使用外部实验参数 theta_T

# ============================================================



    if temp > theta_T:

        return action_set[np.argmin(total_depth)]

    else:

        return None



#---------------修改完毕--------------------------------



def action_to_pos(drone, pos_target, scene):

    """

    根据无人机当前位置和目标位置，计算下一步的最佳动作索引。

    Action Set: ["Go Up", "Go Down", "Turn Left", "Turn Right", "Go Forward", "Go Left", "Go Right"]

    Indices:    0,       1,         2,           3,            4,            5,         6

    """



    # 1. 提取场景边界和步长

    x_min, x_max = scene["x_min"], scene["x_max"]

    y_min, y_max = scene["y_min"], scene["y_max"]

    z_min, z_max = scene["z_min"], scene["z_max"]

    step_x = scene["step_x"]  # 假设水平移动统一使用 step_x

    step_z = scene["step_z"]



    # 辅助函数：检查位置是否在边界内

    def is_valid(pos):

        return (x_min <= pos[0] <= x_max and

                y_min <= pos[1] <= y_max and

                z_min <= pos[2] <= z_max)



    # 获取当前状态

    curr_pos = np.array(drone.pos * [1, -1, -1])

    curr_yaw = drone.ori[2]

    target_pos = np.array(pos_target)



    # ---------------------------------------------------------

    # 第一步：调整高度 (Z轴) - 优先级最高

    # ---------------------------------------------------------

    z_diff = target_pos[2] - curr_pos[2]

    z_threshold = step_z / 2.0  # 设置阈值防止震荡



    if z_diff > z_threshold:

        # 预测向上移动后的位置

        next_pos_up = curr_pos.copy()

        next_pos_up[2] += step_z

        # 只有在不出界的情况下才执行

        if is_valid(next_pos_up):

            return 0  # Go Up



    if z_diff < -z_threshold:

        # 预测向下移动后的位置

        next_pos_down = curr_pos.copy()

        next_pos_down[2] -= step_z

        if is_valid(next_pos_down):

            return 1  # Go Down



    # ---------------------------------------------------------

    # 第二步：转向

    # ---------------------------------------------------------



    # 计算目标相对于当前位置的角度

    dx = target_pos[0] - curr_pos[0]

    dy = target_pos[1] - curr_pos[1]

    target_angle = math.degrees(math.atan2(dy, dx))



    # 计算角度差 (-180 到 180)

    angle_diff = target_angle + curr_yaw

    while angle_diff > 180: angle_diff -= 360

    while angle_diff < -180: angle_diff += 360



    if angle_diff < -45:

        return 3  # Turn Left

    if angle_diff > 45:

        return 2  # Turn Right





    # ---------------------------------------------------------

    # 第三步：移动

    # ---------------------------------------------------------



    # 计算当前水平距离

    curr_dist_xy = np.linalg.norm(target_pos[:2] - curr_pos[:2])



    # 如果已经非常接近目标（小于半个步长），则停止或微调（此处可视需求返回特定动作，这里假设继续尝试对齐）

    if curr_dist_xy < step_x / 2.0:

        # 如果高度也对齐了，理论上应该悬停，这里默认不做动作或继续保持

        return 4



    moves = []



    # 定义移动逻辑 (需与 Drone 类中的数学逻辑保持一致)

    # Forward

    dx_f = step_x * np.cos(math.radians(curr_yaw))

    dy_f = step_x * np.sin(math.radians(curr_yaw))

    pos_fwd = curr_pos.copy()

    pos_fwd[0] += dx_f

    pos_fwd[1] -= dy_f

    moves.append({'action': 4, 'pos': pos_fwd, 'dist': np.linalg.norm(target_pos[:2] - pos_fwd[:2])})



    # Left (Yaw - 90)

    dx_l = step_x * np.cos(math.radians(curr_yaw - 90))

    dy_l = step_x * np.sin(math.radians(curr_yaw - 90))

    pos_left = curr_pos.copy()

    pos_left[0] += dx_l

    pos_left[1] -= dy_l

    moves.append({'action': 5, 'pos': pos_left, 'dist': np.linalg.norm(target_pos[:2] - pos_left[:2])})



    # Right (Yaw + 90)

    dx_r = step_x * np.cos(math.radians(curr_yaw + 90))

    dy_r = step_x * np.sin(math.radians(curr_yaw + 90))

    pos_right = curr_pos.copy()

    pos_right[0] += dx_r

    pos_right[1] -= dy_r

    moves.append({'action': 6, 'pos': pos_right, 'dist': np.linalg.norm(target_pos[:2] - pos_right[:2])})



    # 过滤掉会导致出界的动作

    valid_moves = [m for m in moves if is_valid(m['pos'])]



    if valid_moves:

        # 找出距离目标最近的动作

        best_move = min(valid_moves, key=lambda x: x['dist'])



        # 如果最佳移动能显著减小距离（比当前距离更近），则执行该移动

        # 这里的 "显著" 可以只是简单的 < curr_dist_xy，或者加一点余量

        if best_move['dist'] < curr_dist_xy:

            return best_move['action']







def calculate_angle(pos_drone, pos_target):

    delta_x = pos_target[0] - pos_drone[0]

    delta_y = pos_target[1] - pos_drone[1]

    angle = math.degrees(math.atan2(delta_y, delta_x))  # 计算目标点的角度

    return angle % 360  # 确保角度在0-360度之间





def plot_drone_path(drone_positions):

    """

    绘制无人机移动路径

    """

    # 提取x, y, z坐标

    x_coords = [pos[0] for pos in drone_positions]

    y_coords = [pos[1] for pos in drone_positions]

    z_coords = [pos[2] for pos in drone_positions]



    # 创建3D图形

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d')



    # 绘制路径线

    ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Drone Path')



    # 绘制起点和终点

    ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start', marker='^')

    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End', marker='v')



    # 标记每个位置点

    ax.scatter(x_coords, y_coords, z_coords, color='purple', s=50, alpha=0.6)



    # 设置标签和标题

    ax.set_xlabel('X')

    ax.set_ylabel('Y')

    ax.set_zlabel('Z')

    ax.set_title('Drone Movement Path')

    ax.legend()



    # 添加网格

    ax.grid(True)



    plt.tight_layout()

    plt.show()





def get_action_label():

    while True:

        try:

            # 获取用户输入

            action_label = int(input("请输入动作标签 (0-6): "))  # Prompt for input

            if 0 <= action_label <= 6:  # Validate the input

                return action_label

            else:

                print("无效输入，请输入0到6之间的数字。")  # Invalid input message

        except ValueError:

            print("无效输入，请输入一个数字。")  # Handle non-integer input







# ============================================================

# 2026-10-03 新增：导航历史 / 循环检测

#

# 为什么需要：

# Qwen 每个 Step 原本只看到“当前 RGB + 目标图”，不知道刚刚执行过什么。

# Task21 中已经观察到大量：

#   Go Left -> Go Right -> Go Left -> Go Right

# 以及回到两步之前位置的往返现象。

#

# 返回：

#   history_hint: 给 VLM 的短历史提示

#   avoid_action: 若确认出现往返，建议本轮临时禁止的“撤销动作”

#

# 注意：

# - 这里只处理明显的短周期振荡，不代替 Cognitive/Uncertainty Adviser。

# - 真正边界合法性仍由 main.py 的硬检查负责。

# ============================================================

def build_navigation_history_hint(

    action_history=None,

    position_history=None,

    max_actions=8,

    position_eps=1e-3,

):

    action_history = list(action_history or [])

    position_history = list(position_history or [])



    recent_actions = action_history[-max_actions:]



    opposite_action = {

        "Go Left": "Go Right",

        "Go Right": "Go Left",

        "Turn Left": "Turn Right",

        "Turn Right": "Turn Left",

        "Go Up": "Go Down",

        "Go Down": "Go Up",

    }



    loop_detected = False

    loop_reasons = []

    avoid_action = None



    # 1) 四步交替振荡：A B A B，并且 A/B 互为反向

    if len(action_history) >= 4:

        a, b, c, d = action_history[-4:]



        if (

            a == c

            and b == d

            and a != b

            and opposite_action.get(a) == b

        ):

            loop_detected = True

            loop_reasons.append(

                f"alternating actions detected: {action_history[-4:]}"

            )

            avoid_action = opposite_action.get(d)



    # 2) 两步回到原位：当前位置 ~= 两个动作前的位置

    if len(position_history) >= 3:

        try:

            current_pos = np.asarray(position_history[-1], dtype=float)

            two_steps_ago = np.asarray(position_history[-3], dtype=float)



            if np.linalg.norm(current_pos - two_steps_ago) <= position_eps:

                loop_detected = True

                loop_reasons.append(

                    "current position is approximately the same as two steps ago"

                )



                if action_history:

                    avoid_action = opposite_action.get(

                        action_history[-1],

                        avoid_action,

                    )

        except Exception:

            pass



    # 3) 最近 6 个位置中，同一位置多次出现

    if len(position_history) >= 4:

        try:

            recent_positions = [

                np.asarray(p, dtype=float)

                for p in position_history[-6:]

            ]

            current_pos = recent_positions[-1]



            repeat_count = sum(

                np.linalg.norm(p - current_pos) <= position_eps

                for p in recent_positions

            )



            if repeat_count >= 3:

                loop_detected = True

                loop_reasons.append(

                    f"same area revisited {repeat_count} times recently"

                )

        except Exception:

            pass



    lines = []



    if recent_actions:

        lines.append(

            f"Recent executed actions: {recent_actions}."

        )



    if len(position_history) >= 2:

        try:

            compact_positions = [

                np.round(

                    np.asarray(p, dtype=float),

                    2

                ).tolist()

                for p in position_history[-5:]

            ]

            lines.append(

                f"Recent map positions (oldest -> newest): "

                f"{compact_positions}."

            )

        except Exception:

            pass



    if loop_detected:

        lines.append(

            "Navigation loop warning: "

            + "; ".join(loop_reasons)

            + "."

        )

        lines.append(

            "Do NOT immediately undo the previous movement. "

            "Break the oscillation by choosing a different legal direction "

            "or changing viewpoint before translating again."

        )

    else:

        lines.append(

            "Use the recent history to avoid revisiting the same area "

            "without gaining new visual information."

        )



    return "\n".join(lines), avoid_action





# ============================================================

# 2026-10-02 修改

# 新增 valid_actions=None。

# 如果 main.py 传入当前 Scene 下的合法动作集合，就把它作为硬约束提示给 LLM。

# 真正执行前 main.py 还会再次检查，形成“双层保护”。

# ============================================================

def get_action_from_llm(

    adviser_cognitive_map,

    adviser_uncertainty_map,

    target_text,

    target_image_path,

    rgb_path,

    Attraction_Value,

    valid_actions=None,

    action_history=None,

    position_history=None,

):

    max_retries = 3

    # 记录本轮已经被拒绝的动作，

    # 避免 temperature=0 时连续返回完全相同动作。

    rejected_actions = []



    # ============================================================

    # 2026-10-03 新增：在本轮决策前分析导航历史。

    # ============================================================

    history_hint, history_avoid_action = build_navigation_history_hint(

        action_history=action_history,

        position_history=position_history,

    )



    for attempt in range(max_retries):

        try:

            valid_actions_for_prompt = None



            # 构建 prompt

            if adviser_uncertainty_map:



                if adviser_cognitive_map:

                    prompt = f"""  



                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:

                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.

                                -Image_Object: An image of the object you are searching for, which is the second image.

                                -Text_Object: The object you are searching for is {target_text}.

                                Guidelines:

                                -Exploitation advice: Select action "{adviser_cognitive_map}" helps you to approach the target with a probability of {Attraction_Value * 100}%.

                                -Exploration advice: Select action "{adviser_uncertainty_map}" helps you explore the surrounding environment, which is more important.

                                Select your action follow the guidelines above. Only return the name of the action you selected.

                                """

                else:

                    prompt = f"""  



                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:

                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.

                                -Image_Object: An image of the object you are searching for, which is the second image.

                                -Text_Object: The object you are searching for is {target_text}.

                                Guidelines:

                                -Exploration advice: Select action "{adviser_uncertainty_map}" helps you search for the target.

                                -You can follow the exploration advice or you can also follow your own ideas.

                                Select your action follow the guidelines above. Only return the name of the action you selected.

                                """

            else:

                if adviser_cognitive_map:

                    prompt = f"""  

                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:

                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.

                                -Image_Object: An image of the object you are searching for, which is the second image.

                                -Text_Object: The object you are searching for is {target_text}.

                                Guidelines:

                                -Exploitation advice: Select action "{adviser_cognitive_map}" helps you to approach the target  with a probability of {Attraction_Value * 100}%.

                                -You can follow the exploitation advice or you can also follow your own ideas.

                                -Only return the name of the action you selected.

                                """

                else:

                    prompt = f"""  

                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:

                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.

                                -Image_Object: An image of the object you are searching for, which is the second image.

                                -Text_Object: The object you are searching for is {target_text}.

                                Guidelines:

                                -Follow your own ideas.

                                -Choose exactly one action from the currently legal action set provided below.

                                -Only return the name of the action you selected.

                                """





            # ============================================================

            # 2026-10-03 新增：最近动作 / 位置历史。

            # ============================================================

            if history_hint:

                prompt += (

                    "\nNavigation history:\n"

                    + history_hint

                    + "\n"

                )



            # ============================================================

            # 2026-10-02 新增

            # 把当前 Scene 下不会越界的动作告诉 LLM。

            # 这只是 Prompt 层约束；main.py 仍会在执行前做硬边界检查。

            # ============================================================

            if valid_actions:

                valid_actions_for_prompt = list(valid_actions)

                # 修改2：不再额外限制为 5 个平面动作。
                # 当前动作集合只由 Scene 物理边界决定，Go Up / Go Down 只要不越界也允许。



                # ========================================================

                # 修改3：关闭短周期循环的硬删除，仅保留导航历史提示

                #

                # 循环检测信息仍会作为 Navigation history 提示给模型。

                # 这里只关闭“从合法动作集合硬删除动作”的行为，避免与 Adviser 冲突。

                # ========================================================

                # if (

                #     history_avoid_action

                #     and history_avoid_action in valid_actions_for_prompt

                #     and len(valid_actions_for_prompt) > 1

                # ):

                #     filtered_actions = [

                #         action

                #         for action in valid_actions_for_prompt

                #         if action != history_avoid_action

                #     ]



                #     if filtered_actions:

                #         print(

                #             "[LOOP GUARD] detected oscillation | "

                #             f"temporarily avoid={history_avoid_action} | "

                #             f"legal={valid_actions_for_prompt} -> "

                #             f"{filtered_actions}"

                #         )

                #         valid_actions_for_prompt = filtered_actions



                if valid_actions_for_prompt:

                    prompt += (

                        "\nBoundary constraint:\n"

                        "- The drone must stay inside the current search scene.\n"

                        "- You MUST choose one action only from this currently "

                        f"legal set: {valid_actions_for_prompt}.\n"

                        "- Do not choose an action outside this legal set.\n"

                    )



                if rejected_actions:

                    prompt += (

                        "\nPrevious invalid choices:\n"

                        f"- These actions were already rejected: {rejected_actions}\n"

                        "- Do NOT repeat any rejected action.\n"

                        "- Choose exactly one action from the currently legal set.\n"

                    )



            # 提取返回的动作

            print(prompt)



            # ============================================================

            # 原作者代码（保留，不执行）

            # 原作者这里的图片顺序为：

            #   第1张：目标参考图

            #   第2张：当前无人机 RGB 图

            # 但上面的 Prompt 描述恰好相反：

            #   第1张应为当前 RGB 图，第2张应为目标参考图

            # 同时，原作者直接使用 `chosen_action in action_set` 做严格匹配，

            # 当模型返回 "Turn Left."、"Turn Left\n" 等带标点/空白的结果时会判定失败。

            #

            # chosen_action = chat_with_llm_images(prompt, [("./" + target_image_path), rgb_path])

            # print(chosen_action)

            #

            # # 验证动作是否在动作集中

            # if chosen_action in action_set:

            #     # 返回动作在列表中的索引

            #     return action_set.index(chosen_action)

            # else:

            #     print(f"Attempt {attempt + 1}: Invalid action chosen. Retrying...")

            #     time.sleep(1)  # 短暂延迟后重试

            # ============================================================



            # ============================================================

            # 2026-09-28 修改

            # 修改内容：

            # 1. 修正多模态图片顺序，使其与 Prompt 保持一致：

            #       第1张 = 当前无人机 RGB 图

            #       第2张 = 目标参考图

            # 2. 保留模型原始输出用于调试。

            # 3. 对模型返回动作进行轻量标准化：

            #       去除首尾空白、引号、反引号以及末尾常见标点，

            #       并忽略大小写进行匹配。

            #    这样 "Turn Left." / "turn left" / "`Turn Left`" 都可以正确识别。

            # ============================================================



            # ============================================================

            # 2026-10-02 新增：Qwen 单次请求测速

            # 作用：记录每次 API 尝试耗时和返回值，不改变原有重试逻辑。

            # ============================================================

            qwen_attempt_start = time.perf_counter()

            print(f"[QWEN TEST] Attempt {attempt + 1}/{max_retries} start")



            chosen_action = chat_with_llm_images(

                prompt,

                [rgb_path, target_image_path]

            )



            qwen_attempt_time = time.perf_counter() - qwen_attempt_start

            print(

                f"[QWEN TEST] Attempt {attempt + 1}/{max_retries} "

                f"time = {qwen_attempt_time:.2f}s"

            )

            print("[QWEN TEST] Raw result:", repr(chosen_action))



            # 打印模型原始输出，便于排查模型是否返回了多余字符

            print("LLM raw action:", repr(chosen_action))



            # 对返回结果进行标准化

            chosen_action_clean = chosen_action.strip()

            chosen_action_clean = chosen_action_clean.strip("`'\"* ")

            chosen_action_clean = chosen_action_clean.rstrip(".,;:!?")



            # 忽略大小写匹配到标准动作名称

            action_map = {action.lower(): action for action in action_set}

            chosen_action_normalized = action_map.get(chosen_action_clean.lower())



            print("LLM normalized action:", chosen_action_normalized)



            # ============================================================

            # 2026-10-03 修改：

            # 除了必须属于 action_set，还必须属于本轮 Prompt 的合法动作集合。

            # Stop 单独允许；移动动作仍需通过 Scene 物理边界约束。

            # ============================================================

            # 修改3：真正的 Stop 允许通过；其他动作必须属于当前 Scene 合法动作集合。
            action_is_allowed = (
                chosen_action_normalized is not None
                and (
                    chosen_action_normalized == "Stop"
                    or not valid_actions_for_prompt
                    or chosen_action_normalized in valid_actions_for_prompt
                )
            )



            if action_is_allowed:

                return action_set.index(chosen_action_normalized)

            else:

                print(

                    f"Attempt {attempt + 1}: Invalid action chosen: "

                    f"{repr(chosen_action)}. Retrying..."

                )



                if (

                    chosen_action_normalized is not None

                    and chosen_action_normalized not in rejected_actions

                ):
                    rejected_actions.append(chosen_action_normalized)

                time.sleep(1)  # 短暂延迟后重试



        except Exception as e:

            # ============================================================

            # 2026-10-02 修改：输出失败尝试信息

            # 作用：判断是否因超时/网络/API 异常触发重试。

            # ============================================================

            print(

                f"[QWEN TEST] Attempt {attempt + 1}/{max_retries} failed:",

                repr(e)

            )

            time.sleep(2)  # 错误后延迟重试



        # 修改4：如果重试仍失败，返回 None，交给 main.py 安全恢复

    # print("Failed to get a valid action after multiple attempts. Defaulting to 'Stop'.")

    # return action_set.index('Stop')

    # 3次失败后不要返回 Stop

    print(

        "Failed to get a valid action after multiple attempts. "

        "Returning None for safe recovery."

    )



    return None







def find_max_cluster_center(cognitive_map, eps=1.0, min_samples=3):

    """

    从cognitive_map中提取第四维值最大的点，并对这些点进行聚类，返回最大簇的聚类中心。



    参数:

        cognitive_map: np.array, 形状为 (N, 4)

        eps: DBSCAN的半径参数

        min_samples: DBSCAN的最小样本数参数



    返回:

        max_value: 第四维的最大值

        cluster_center: 最大簇的中心坐标 (x, y, z)

    """



    # 1. 输入校验

    if cognitive_map is None or len(cognitive_map) == 0:

        return None, None



    # 2. 找出第四维的最大值

    max_value = np.max(cognitive_map[:, 3])



    # ============================================================

    # 2026-10-03 性能修复：全零 Cognitive Map 直接返回。

    #

    # 原逻辑在 max_value == 0 时，会把“整张地图的所有点”都当成

    # 最大值点送进 DBSCAN。Scene4 当前约千万级网格点，会导致

    # find_max_cluster_center() 单次耗时数百秒。

    #

    # main.py 只有在 max_value > cognitive_threshold(0.5) 时才会

    # 使用 cluster_center 生成 adviser_cognitive_map；因此 max_value<=0

    # 时 cluster_center 本来就不会影响导航。这里直接返回 None，

    # 不改变有效导航结果，只跳过无意义的超大规模聚类。

    # ============================================================

    if max_value <= 0:

        return max_value, None



    # 3. 找出具有最大值的点 (使用 np.isclose 解决浮点数精度问题)

    # atol 是绝对容差，根据数据量级调整，通常 1e-8 足够

    mask = np.isclose(cognitive_map[:, 3], max_value, atol=1e-8)

    max_value_points = cognitive_map[mask][:, :3]



    # 4. 边界情况处理：如果没有点或点很少

    num_points = len(max_value_points)

    if num_points == 0:

        return max_value, None



    # 如果点数少于聚类要求的最小样本数，直接计算这些点的中心作为结果

    # 避免 DBSCAN 将其标记为噪声导致返回 None

    if num_points < min_samples:

        return max_value, np.mean(max_value_points, axis=0)



    # 5. 使用 DBSCAN 聚类

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    clusters = dbscan.fit_predict(max_value_points)



    # 6. 找出最大簇

    unique_clusters = np.unique(clusters)



    # 过滤掉噪声点 (-1)

    valid_clusters = unique_clusters[unique_clusters != -1]



    if len(valid_clusters) == 0:

        # 情况A: 所有点都被标记为噪声 (-1)

        # 策略: 这种情况下，说明点很分散。

        # 建议直接返回所有最大值点的几何中心作为“最大簇中心”的兜底

        return max_value, np.mean(max_value_points, axis=0)



    # 情况B: 存在有效的簇，寻找包含点数最多的簇

    best_cluster_label = -1

    max_size = -1



    for cluster_label in valid_clusters:

        size = np.sum(clusters == cluster_label)

        if size > max_size:

            max_size = size

            best_cluster_label = cluster_label



    # 7. 计算最大簇的聚类中心

    largest_cluster_points = max_value_points[clusters == best_cluster_label]

    cluster_center = np.mean(largest_cluster_points, axis=0)



    return max_value, cluster_center
