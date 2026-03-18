from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Config:
    """Configuration for the minimal Phase 1 hidden-physics benchmark.
    
    Args:
        timestep: Value for the timestep used by this routine.
        control_substeps: Value for the control substeps used by this routine.
        max_episode_steps: Number of steps to use for max episode.
        workspace_x_min: Value for the workspace x min used by this routine.
        workspace_x_max: Value for the workspace x max used by this routine.
        workspace_y_min: Value for the workspace y min used by this routine.
        workspace_y_max: Value for the workspace y max used by this routine.
        workspace_z_min: Value for the workspace z min used by this routine.
        workspace_z_max: Value for the workspace z max used by this routine.
        home_x: Value for the home x used by this routine.
        home_y: Value for the home y used by this routine.
        home_z: Value for the home z used by this routine.
        object_half_extent: Value for the object half extent used by this routine.
        default_object_x: Value for the default object x used by this routine.
        default_object_y: Value for the default object y used by this routine.
        default_object_z: Value for the default object z used by this routine.
        target_x: Value for the target x used by this routine.
        target_y: Value for the target y used by this routine.
        target_radius: Value for the target radius used by this routine.
        target_random_x_min: Lower bound for randomized target x coordinates.
        target_random_x_max: Upper bound for randomized target x coordinates.
        target_random_y_min: Lower bound for randomized target y coordinates.
        target_random_y_max: Upper bound for randomized target y coordinates.
        target_random_min_distance: Minimum XY distance between object start and randomized target.
        open_aperture: Value for the open aperture used by this routine.
        default_close_aperture: Value for the default close aperture used by this routine.
        max_aperture: Value for the max aperture used by this routine.
        action_delta_limit: Value for the action delta limit used by this routine.
        object_masses: Value for the object masses used by this routine.
        object_frictions: Value for the object frictions used by this routine.
        object_stiffnesses: Value for the object stiffnesses used by this routine.
        settle_steps: Number of steps to use for settle.
        move_tolerance: Value for the move tolerance used by this routine.
        hold_steps: Number of steps to use for hold.
        lift_height: Height used for lift.
        place_height: Height used for place.
    """

    timestep: float = 0.002
    control_substeps: int = 20
    max_episode_steps: int = 240
    workspace_x_min: float = -0.18
    workspace_x_max: float = 0.22
    workspace_y_min: float = -0.20
    workspace_y_max: float = 0.20
    workspace_z_min: float = 0.015
    workspace_z_max: float = 0.22
    home_x: float = 0.0
    home_y: float = 0.0
    home_z: float = 0.16
    object_half_extent: float = 0.015
    default_object_x: float = 0.0
    default_object_y: float = -0.07
    default_object_z: float = 0.015
    target_x: float = 0.18
    target_y: float = 0.12
    target_radius: float = 0.05
    target_random_x_min: float = 0.10
    target_random_x_max: float = 0.20
    target_random_y_min: float = -0.16
    target_random_y_max: float = 0.16
    target_random_min_distance: float = 0.10
    open_aperture: float = 0.0
    default_close_aperture: float = 0.014
    max_aperture: float = 0.022
    action_delta_limit: float = 0.02
    object_masses: tuple[float, ...] = (0.06, 0.12, 0.18)
    object_frictions: tuple[float, ...] = (0.35, 0.60, 0.95)
    object_stiffnesses: tuple[float, ...] = (0.55, 0.75, 0.95)
    settle_steps: int = 120
    move_tolerance: float = 0.003
    hold_steps: int = 30
    lift_height: float = 0.12
    place_height: float = 0.05
    adaptation_success_horizon_steps: int = 220
