#! /usr/bin/env python

import rospy
from collecting_with_dorna.msg import Coordinates
from dorna import Dorna
from calculate_joint_values import JointValue

class Subscriber:
    def __init__(self):
        rospy.init_node("coordinates_subscriber")
        self.coordinates_sub = rospy.Subscriber("/coordinates", Coordinates, self.callback)
        self.coordinates_msg = Coordinates()

        self.robot = Dorna("/home/lennart/dorna/dorna/my_config.yaml")
        self.robot.connect()

    def home(self):
        proceed = False

        while not proceed:
            _input = input("Home Dorna? (y/n)")

            if _input == "y":
                proceed = True

            elif _input == "n":
                self.terminate()
                break

        if proceed:
            self.robot.home(["j0", "j1", "j2", "j3"])
            proceed = False

            while not proceed:
                _input = input("j3 auf 0 setzen? (y/n)")

                if _input == "y":
                    proceed = True

                elif _input == "n":
                    break

            if proceed:
                self.set_joint_j3_0()

    def set_joint_j3_0(self):
        self.robot.set_joint({"j3": 0})
        self.robot.set_joint({"j4": 0})

    def callback(self, msg):
        self.coordinates_msg = msg

    def collect_balls(self):
        container_coordinate = self.coordinates_msg.container_coordinates

        if len(container_coordinate) > 0:
            x_c = container_coordinate[0]
            y_c = container_coordinate[1]
            z_c = 0

            print("Behälterposition: ", x_c, y_c)

        ball_coordinate = self.coordinates_msg.ball_coordinates

        balls_array = []

        if len(ball_coordinate) > 0:
            for i in range(0, len(ball_coordinate), 2):

                x_b, y_b = ball_coordinate[i], ball_coordinate[i + 1]
                z_b = 0

                print("Ballposition: ", x_b, y_b)

                five_above_ball = JointValue(x_b, y_b, z_b + 5)
                one_above_ball = JointValue(x_b, y_b, z_b + 1)

                try:
                    five_above_ball_result = five_above_ball.calc_joint_values()
                    one_above_ball_result = one_above_ball.calc_joint_values()
                except ValueError:
                    print("Ball außerhalb des Bereiches")
                    continue

                five_above_ball_array = self.get_joints(five_above_ball_result)
                one_above_ball_array = self.get_joints(one_above_ball_result)

                balls_dict = {"1.": five_above_ball_array, "2.": one_above_ball_array, "3.": five_above_ball_array}

                balls_array.append(balls_dict)



            ten_above_container = JointValue(x_c, y_c, z_c + 10)

            try:
                ten_above_container_result = ten_above_container.calc_joint_values()
            except ValueError:
                print("Behälter außerhalb des Bereiches")

            ten_above_container_array = self.get_joints(ten_above_container_result)

            for i in balls_array:
                first = i.get("1.")
                second = i.get("2.")
                third = i.get("3.")

                self.open_gripper()
                self.move_dorna(first)
                self.move_dorna(second)
                self.close_gripper()
                self.move_dorna(third)
                self.move_dorna(ten_above_container_array)
                self.open_gripper()

        else:
            print("Konnte keine Bälle orten, versuche erneut")
            rospy.sleep(1)
            self.collect_balls()

    def terminate(self):
        self.robot.terminate()

    def close_gripper(self):
        self.robot.set_io({"servo": 675})

    def open_gripper(self):
        self.robot.set_io({"servo": 0})

    def move_dorna(self, position):
        self.robot.move({"movement": 0, "path": "joint", "joint": position})

    def get_joints(self, height):
        j0 = height.get("j0")
        j1 = height.get("j1")
        j2 = height.get("j2")
        j3 = height.get("j3")

        return [j0, j1, j2, j3, 0]

if __name__ == '__main__':
    dorna = Subscriber()
    dorna.home()
    try_again = True
    while try_again:
        dorna.collect_balls()
        while True:
            _input = input("Nochmal? (y/n)")

            if _input == "n":
                try_again = False
                break

            elif _input == "y":
                break

    dorna.terminate()