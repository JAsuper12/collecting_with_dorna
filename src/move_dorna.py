from dorna import Dorna
from calculate_joint_values import JointValue, Input


class MoveDorna:
    def __init__(self):
        self.robot = Dorna("/home/lennart/dorna/dorna/my_config.yaml")
        self.robot.connect()

    def home(self):
        proceed = False

        while not proceed:
            _input = input("Home Dorna? (y/n)")

            if _input == "y":
                proceed = True

            elif _input == "n":
                break

        if proceed:
            self.robot.home(["j0", "j1", "j2", "j3"])
            print(self.robot.homed())
            print(self.robot.position())
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

    def move_to_position(self):
        i = Input()
        output = i.input_values()
        x = output.get("x")
        y = output.get("y")
        z = output.get("z")

        position = JointValue(x, y, z)
        position_result = position.calc_joint_values()
        position_array = self.get_joints(position_result)

        print(position_array)
        proceed = True

        while proceed:
            _input = input("Bewegung ausführen? (y/n)")

            if _input == "n":
                proceed = False

            elif _input == "y":
                break

        if proceed:
            self.move_dorna(position_array)

    def collect_balls(self):
        n = eval(input("Anzahl der Bälle: "))
        balls_array = []
        for j in range(n):
            print("Position des", j + 1, ". Balls:")
            i = Input()
            output = i.input_values()
            x = output.get("x")
            y = output.get("y")
            z = output.get("z")

            five_over_ball = JointValue(x, y, z + 5)
            one_over_ball = JointValue(x, y, z + 1)

            five_over_ball_result = five_over_ball.calc_joint_values()
            one_over_ball_result = one_over_ball.calc_joint_values()

            five_over_ball_array = self.get_joints(five_over_ball_result)
            one_over_ball_array = self.get_joints(one_over_ball_result)

            balls_dict = {"1.": five_over_ball_array, "2.": one_over_ball_array, "3.": five_over_ball_array}

            balls_array.append(balls_dict)

        print("Position des Behälters: ")
        i = Input()
        output = i.input_values()
        x = output.get("x")
        y = output.get("y")
        z = output.get("z")

        ten_over_container = JointValue(x, y, z + 10)
        ten_over_container_result = ten_over_container.calc_joint_values()
        ten_over_container_array = self.get_joints(ten_over_container_result)

        print(balls_array)
        balls_array_len = len(balls_array)
        for x in range(balls_array_len):
            print(balls_array[x])
        print(ten_over_container_array)
        proceed = True

        while proceed:
            _input = input("Bewegung ausführen? (y/n)")

            if _input == "n":
                proceed = False

            elif _input == "y":
                break

        if proceed:
            for i in range(n):
                first = balls_array[i].get("1.")
                second = balls_array[i].get("2.")
                third = balls_array[i].get("3.")

                self.open_gripper()
                self.move_dorna(first)
                self.move_dorna(second)
                self.close_gripper()
                self.move_dorna(third)
                self.move_dorna(ten_over_container_array)
                self.open_gripper()

            print("Bewegung ausgeführt.")

    def terminate(self):
        self.robot.terminate()

    def close_gripper(self):
        self.robot.set_io({"servo": 675})

    def open_gripper(self):
        self.robot.set_io({"servo": 0})

    def get_joints(self, height):
        j0 = height.get("j0")
        j1 = height.get("j1")
        j2 = height.get("j2")
        j3 = height.get("j3")

        return [j0, j1, j2, j3, 0]

    def move_dorna(self, position):
        self.robot.move({"movement": 0, "path": "joint", "joint": position})

    def zero(self):
        self.robot.move({"movement": 0, "path": "joint", "joint": [0, 0, 0, 0, 0]})


if __name__ == "__main__":
    dorna = MoveDorna()
    dorna.home()
    proceed = False
    repeat = True

    while repeat:
        try:
            _input = input("(p): Position eingeben \n(c): Bälle einsammeln \n(z): zugreifen \n(a): Greifer oeffnen "
                           "\n(o): In die Nullposition fahren: ")
        except NameError:
            continue
        except SyntaxError:
            continue

        if _input == "p":
            dorna.move_to_position()
            proceed = True

        elif _input == "c":
            dorna.collect_balls()
            proceed = True

        elif _input == "z":
            dorna.close_gripper()
            proceed = True

        elif _input == "a":
            dorna.open_gripper()
            proceed = True

        elif _input == "o":
            dorna.zero()
            proceed = True

        else:
            proceed = False

        while proceed:
            _input = input("Fortsetzen? (y/n)")

            if _input == "n":
                repeat = False
                break

            elif _input == "y":
                repeat = True
                break

    dorna.terminate()
