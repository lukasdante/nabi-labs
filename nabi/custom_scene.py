import argparse
import os
from omni.isaac.lab.app import AppLauncher

# add arguments
parser = argparse.ArgumentParser(description="Custom scene set-up for Arctos RL Environment.")

# add custom arguments
parser.add_argument("--test", type=str, default="Some random text",
                    help="This is a sample argument.")
parser.add_argument("--size", type=tuple[float,float,float], default=None, 
                    help="Size of the table.")
parser.add_argument("--num_origins", type=int, default=None, help="Number of origins.")

# append AppLauncher cli args and parse the arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app with the arguments
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# get current working directory
cwd = os.getcwd()

# once the app is launched, import the necessary modules
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, ArticulationCfg, Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import torch
import numpy as np

def create_table(size):
    """ Creates a table in the scene. """
    
    # add size argument
    if args_cli.size:
        size=args_cli.size

    # configure the table object
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Table",

        # spawn a cube with rigid, mass, collision, and visual properties
        
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        )
    )

    # create the table object then return scene information
    table = RigidObject(cfg=cube_cfg)
    scene_entities = {"table": table}

    return scene_entities
    
def create_glass():
    """ Creates a glass in the scene. """

    visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=0.5, thin_walled=True, color=(0.5, 0.5, 0.5))
    visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)

    # configure the glass object
    glass_cfg = RigidObjectCfg(
        prim_path="/World/Glass",

        # spawn a cube with rigid, mass, collision, and visual properties
        
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    )

    # create the glass object then return scene information
    glass = RigidObject(cfg=glass_cfg)

    sim_utils.bind_visual_material("/World/Glass", "/World/Looks/glassMaterial")

    scene_entities = {"glass": glass}

    return scene_entities

def create_cylinder_glass(index, translation=(1.0, 1.0, 1.0)):
    """ Creates a glass in the scene. """

    # configure the glass object
    glass_cfg = RigidObjectCfg(
        prim_path=f"/World/Glass{index}",

        # spawn a cube with rigid, mass, collision, and visual properties
        
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    )

    

    glass_cfg.init_state.pos = translation

    glass = RigidObject(cfg=glass_cfg)

    sim_utils.bind_visual_material(f"/World/Glass{index}", "/World/Looks/glassMaterial")

    scene_entities = {"glass": glass}

    return scene_entities

    # create the glass object then


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""


    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()

def design_scene():
    """ Initializes the scene. """
    NUM_ORIGINS = 1

    if args_cli.num_origins:
        NUM_ORIGINS = args_cli.num_origins

    origins = define_origins(num_origins=NUM_ORIGINS, spacing=2.0)

    SIZE = (1.25, 1.25, 0.5)
    # ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # load usd files
    usd_file_path = os.path.join(cwd, "nabi/arctos.usd")

    visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=1.47, thin_walled=True, frosting_roughness=0.1, glass_color=(0.5, 0.5, 0.5))
    visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)
    
    # spawning the usd file
    for i in range(NUM_ORIGINS):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=(origins[i]))
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
        cfg.func(f"/World/Origin{i}/Table", cfg, translation=(0.0, 0.0 , 0.8))
        cfg = sim_utils.UsdFileCfg(usd_path=usd_file_path, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
        # configuring the articulation
        initial_joint_positions = {"lower_base": 0.0, "middle_lower": 0.0, "upper_middle": 0.0,
                                "wrist_upper": 0.0, "inner_wrist": 0.0, "outer_inner": 0.0,
                                "slide_left": 0.0, "slide_right": 0.0}
        ARM_CFG = ArticulationCfg(spawn=cfg, prim_path=f"/World/Origin{i}/arctos", 
                                init_state=ArticulationCfg.InitialStateCfg(
                                    pos=(0.0, 0.0, 0.8), joint_pos=initial_joint_positions),
                                )
        arm = Articulation(cfg=ARM_CFG)
        glass_entity = create_cylinder_glass(i, translation=(0.5 + origins[i][0], 0 + origins[i][1], 0.8 + origins[i][2]))
    
    # create a table in the scene
    # table_entity = create_table(size=SIZE)
       
    # TODO: You can load other entities in the scenes here.


    # adding the table to the scene entities
    scene_entities = {}
    scene_entities.update({"arm": arm})
    # scene_entities.update(table_entity)
    # TODO: You can update the scene entities with other entities here.

    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """ Runs the simulator. """

    # extract scene entities
    # table_object = entities["table"]
    robot = entities["arm"]

    # define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    desired_joint_pos = torch.tensor([[0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # simulate physics


    while simulation_app.is_running():
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.step()

def main():
    """ Main function. """

    # load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # set main camera
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])

    # design the scene
    scene_entities = design_scene()

    # play the simulator
    sim.reset()
    print("[INFO]: TO THE MOOON!!!")

    # run the simulator
    run_simulator(sim, scene_entities)

if __name__ == "__main__":
    main()
    simulation_app.close()