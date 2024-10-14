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
import torch


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

    # configure the glass object
    glass_cfg = RigidObjectCfg(
        prim_path="/World/Glass",

        # spawn a cube with rigid, mass, collision, and visual properties
        
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=None,
        )
    )

    # create the glass object then return scene information
    glass = RigidObject(cfg=glass_cfg)
    
    scene_entities = {"glass": glass}

    return scene_entities

    

def design_scene():
    """ Initializes the scene. """

    SIZE = (1.25, 1.25, 0.5)
    # ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # initialize scene entities
    scene_entities = {}
    
    # create a table in the scene
    table_entity = create_table(size=SIZE)
    glass_entity = create_glass()   
    # TODO: You can load other entities in the scenes here.

    # load usd files
    usd_file_path = os.path.join(cwd, "nabi/arctos.usd")
    
    # spawning the usd file
    spawn = sim_utils.UsdFileCfg(usd_path=usd_file_path, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6), metallic=0.2, roughness=0.5))
    
    # configuring the articulation
    ARM_CFG = ArticulationCfg(spawn=spawn, prim_path="/World/Arm")
    arm = Articulation(cfg=ARM_CFG)

    # adding the table to the scene entities
    scene_entities.update(table_entity)
    scene_entities.update(glass_entity)
    scene_entities.update({"arm": arm})
    # TODO: You can update the scene entities with other entities here.

    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """ Runs the simulator. """

    # extract scene entities
    table_object = entities["table"]
    glass_object = entities["glass"]
    robot = entities["arm"]

    # define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # simulate physics
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            print(root_state)
            # root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        
        table_object.write_data_to_sim()
        glass_object.write_data_to_sim()

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)

def main():
    """ Main function. """

    # load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # set main camera
    sim.set_camera_view(eye=[2.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])

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