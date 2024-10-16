#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given dummy.
This node publishes and subscribes the following topics:

    PUBLICATIONS            SUBSCRIPTIONS
    /drone_command           /whycon/poses
    /pid_error               /throttle_pid
                             /pitch_pid
                             /roll_pid
'''

# Importing the required libraries

from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
import rclpy
from rclpy.node import Node


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # initializing ros node with name pico_controller

        # This corresponds to your current position of the drone. This value must be updated each time in your whycon callback
        # [x, y, z]
        self.drone_position = [0.0, 0.0, 0.0]

        # [x_setpoint, y_setpoint, z_setpoint]
        self.setpoint = [2, 2, 20]  # whycon marker at the position of the dummy

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Initial setting of Kp, Ki, Kd for [roll, pitch, throttle]
        self.Kp = [0.6, 0.6, 1.2]  # Example initial values
        self.Ki = [0.1, 0.1, 0.2]
        self.Kd = [0.3, 0.3, 0.6]

        # Variables for PID control
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.max_values = [2000, 2000, 2000]  # [roll, pitch, throttle]
        self.min_values = [1000, 1000, 1000]

        # This is the sample time in which you need to run PID
        self.sample_time = 0.060  # in seconds

        # Publishing /drone_command, /pid_error
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

        # Subscribing to /whycon/poses, /throttle_pid, /pitch_pid, roll_pid
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        # Arming the drone
        self.arm()

        # Creating a timer to run the PID function periodically
        self.create_timer(self.sample_time, self.pid)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    # Whycon callback function for current drone position
    def whycon_callback(self, msg):
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z

    # PID tuning for throttle
    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.kp * 0.03  # Adjust scaling as necessary
        self.Ki[2] = alt.ki * 0.008
        self.Kd[2] = alt.kd * 0.6

    # PID tuning for pitch
    def pitch_set_pid(self, pitch):
        self.Kp[1] = pitch.kp * 0.03
        self.Ki[1] = pitch.ki * 0.008
        self.Kd[1] = pitch.kd * 0.6

    # PID tuning for roll
    def roll_set_pid(self, roll):
        self.Kp[0] = roll.kp * 0.03
        self.Ki[0] = roll.ki * 0.008
        self.Kd[0] = roll.kd * 0.6

    def pid(self):
        # PID control for x, y, z (roll, pitch, throttle)
        errors = [self.setpoint[i] - self.drone_position[i] for i in range(3)]

        for i in range(3):  # Iterate for roll, pitch, throttle
            # Proportional term
            p_term = self.Kp[i] * errors[i]

            # Integral term
            self.error_sum[i] += errors[i] * self.sample_time
            i_term = self.Ki[i] * self.error_sum[i]

            # Derivative term
            d_term = self.Kd[i] * (errors[i] - self.prev_error[i]) / self.sample_time

            # PID output
            pid_output = p_term + i_term + d_term

            # Apply PID output to control commands, ensuring values are integers
            if i == 0:  # Roll
                self.cmd.rc_roll = int(1500 + pid_output)
                self.cmd.rc_roll = max(min(self.cmd.rc_roll, self.max_values[0]), self.min_values[0])

            elif i == 1:  # Pitch
                self.cmd.rc_pitch = int(1500 + pid_output)
                self.cmd.rc_pitch = max(min(self.cmd.rc_pitch, self.max_values[1]), self.min_values[1])

            elif i == 2:  # Throttle
                self.cmd.rc_throttle = int(1500 + pid_output)
                self.cmd.rc_throttle = max(min(self.cmd.rc_throttle, self.max_values[2]), self.min_values[2])

            # Update previous error for next iteration
            self.prev_error[i] = errors[i]

        # Publish drone command
        self.command_pub.publish(self.cmd)

        pid_error_msg = PIDError()
        pid_error_msg.roll_error = errors[0]  # Change to correct field name
        pid_error_msg.pitch_error = errors[1]  # Change to correct field name
        pid_error_msg.throttle_error = errors[2]  # Change to correct field name
        self.pid_error_pub.publish(pid_error_msg)


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    rclpy.spin(swift_pico)
    swift_pico.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
