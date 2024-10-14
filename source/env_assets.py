import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Environment set-up for Nabi Cobot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.core.utils.stage as stage_utils


def design_scene():
    arctos_usd = "../arctos/arctos_usd/arctos_revolute/arctos_revolute.usd"
    stage_utils.add_reference_to_stage(
        usd_path=arctos_usd,
        prim_path="/World/Objects/Arctos",
    )




    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(0.0, 0.0, 10.0))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone

    prim_utils.create_prim(
        prim_path="/World/Objects/Cube1",  # Path for the cube in the stage
        prim_type="Cube",                 # Create a Cube primitive
        translation=(0.0, 0.5, 0.0),      # Position the cube at (1, 0, 1)
        scale=(0.5, 0.5, 0.5),            # Scale the cube to 0.5m x 0.5m x 0.5m
    )


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
