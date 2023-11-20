from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Initialize a new launch description.
    ld = LaunchDescription()
    
    # Get path to the params.yaml file.
    simulation_params = Path(get_package_share_directory("mpc"), "config", "simulation_params.yaml")

    # Create new actions to spin up nodes for the pure pursuit node and path
    # file publisher node.
    # pose_publisher_node = Node(
    #     package="pure_pursuit",
    #     executable="pose_publisher_node.py",
    #     parameters=[simulation_params],
    #     # NOTE: For now, not using namespaces until I can spend more time
    #     # figuring out how to manipulate the bridge's namespaces.
    #     remappings=[("odom", "ego_racecar/odom"),
    #                 ("pose", "ego_racecar/pose")]
    #     # arguments=["--ros-args", "--log-level", "debug"],
    # )
    path_file_publisher_node = Node(
        package="pure_pursuit",
        executable="path_publisher_node.py",
        parameters=[simulation_params]
        # remappings=[]
    )
    # pure_pursuit_node = Node(
    #     # namespace="ego_racecar",
    #     package="pure_pursuit",
    #     executable="pure_pursuit_node.py",
    #     parameters=[simulation_params],
    #     remappings=[("pose", "ego_racecar/pose")]
    #     # arguments=["--ros-args", "--log-level", "debug"]
    # )
    mpc_node = Node(
        package="mpc",
        executable="mpc_node.py",
        parameters=[simulation_params],
        remappings=[("odom", "ego_racecar/odom")]
    )

    # Add the launch_ros "Node" actions we created.
    # ld.add_action(pose_publisher_node)
    ld.add_action(path_file_publisher_node)
    # ld.add_action(pure_pursuit_node)
    ld.add_action(mpc_node)

    # Return the newly created launch description.
    return ld