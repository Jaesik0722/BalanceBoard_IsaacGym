"""Evaluation-only physics perturbation (E2: model mismatch).

Applied once at environment creation with a fixed scale, deliberately not via
`apply_randomizations`: a sweep axis has to hold still for the whole episode
and be reproducible from the config alone. The existing randomization block
stays for the combined worst-case condition.

All scales default to 1.0, so training behaviour is untouched.

Mass scaling also scales the inertia tensor by the same factor, which is what
"the robot is 20% heavier" means for a body whose geometry is unchanged.
Scaling mass alone would leave an inertia that no physical object has.
"""


def get_scales(cfg: dict) -> dict:
    eval_cfg = (cfg or {}).get("eval", {}) or {}
    scale_cfg = eval_cfg.get("physicsScale", {}) or {}
    return {
        "robot_mass": float(scale_cfg.get("robotMass", 1.0)),
        "board_mass": float(scale_cfg.get("boardMass", 1.0)),
        "joint_stiffness": float(scale_cfg.get("jointStiffness", 1.0)),
        "joint_damping": float(scale_cfg.get("jointDamping", 1.0)),
    }


NOMINAL_ROLLER_RADIUS = 0.039


def board_asset_path(cfg: dict, default: str = "urdf/board.urdf") -> str:
    eval_cfg = (cfg or {}).get("eval", {}) or {}
    return eval_cfg.get("boardAsset", default) or default


def roller_height_offset(cfg: dict) -> float:
    """How much higher the assembly rides when a different roller is fitted.

    Geometry is read off the shipped asset: the roller sphere's top sits flush
    with the deck's top surface, which fixes the collision origin at 0.015 - r
    and the resting board height at 2r - 0.014. Substituting the nominal
    r = 0.039 reproduces the shipped values (-0.024, 0.064) exactly, so the
    variants are the same assembly with a different ball in it.

    Without the height compensation the board spawns interpenetrating the floor
    (large roller) or dropping onto it (small one), and the sweep would be
    measuring the drop transient rather than the contact curvature.
    """
    eval_cfg = (cfg or {}).get("eval", {}) or {}
    radius = float(eval_cfg.get("rollerRadius", NOMINAL_ROLLER_RADIUS))
    return 2.0 * (radius - NOMINAL_ROLLER_RADIUS)


def any_active(scales: dict) -> bool:
    return any(abs(v - 1.0) > 1e-9 for v in scales.values())


def describe(scales: dict, asset: str) -> str:
    parts = [f"{k}={v:g}" for k, v in scales.items() if abs(v - 1.0) > 1e-9]
    if asset != "urdf/board.urdf":
        parts.append(f"board_asset={asset}")
    return "physics scale: " + (", ".join(parts) if parts else "nominal")


def scale_actor_mass(gym, env_ptr, actor_handle, factor: float):
    """Scale every rigid body's mass and inertia on one actor."""
    if abs(factor - 1.0) <= 1e-9:
        return
    props = gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
    for p in props:
        p.mass = p.mass * factor
        for axis in ("x", "y", "z"):
            row = getattr(p.inertia, axis)
            row.x = row.x * factor
            row.y = row.y * factor
            row.z = row.z * factor
    # recomputeInertia=False: the inertia above is already the scaled one we want.
    gym.set_actor_rigid_body_properties(env_ptr, actor_handle, props, False)
