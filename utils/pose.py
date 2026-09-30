import math

from isaacgym import gymapi


def euler_deg_to_quat(roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> gymapi.Quat:
    """Convert Euler angles in degrees (roll=X, pitch=Y, yaw=Z) to an isaacgym Quat."""
    return gymapi.Quat.from_euler_zyx(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )
