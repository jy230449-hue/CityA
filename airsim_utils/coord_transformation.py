import numpy as np
import airsim


def quaternion2eularian_angles(quat):
    """
    AirSim quaternion -> [pitch, roll, yaw]
    返回单位：弧度
    """
    pry = airsim.to_eularian_angles(quat)
    return np.array([pry[0], pry[1], pry[2]])