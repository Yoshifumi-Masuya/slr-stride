from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("encoder_path"),
        DeclareLaunchArgument("policy_path"),
        DeclareLaunchArgument("armed", default_value="false"),

        Node(
            package="slr_ros2",
            executable="slr_policy_node",
            name="slr_policy_node",
            output="screen",
            parameters=[{
                "encoder_path": LaunchConfiguration("encoder_path"),
                "policy_path": LaunchConfiguration("policy_path"),
            }],
        ),

        Node(
            package="slr_ros2",
            executable="lowcmd_bridge",
            name="lowcmd_bridge",
            output="screen",
            parameters=[{
                "armed": ParameterValue(
                    LaunchConfiguration("armed"), value_type=bool
                ),
            }],
        ),
    ])
