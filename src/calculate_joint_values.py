import math
from cartesian_to_polar import CartToPolar
import numpy


class JointValue:
    def __init__(self, x, y, z):
        # Länge von Link 1, Link 2 und Link 3 im greifenden Zustand
        self.l1 = 20.32
        self.l2 = 15.24
        self.l3 = 17

        self.x = x
        self.y = y
        self.h = z + self.l3 - 20.6

        # umformen der eingegeben x- und y-Koordinaten in Polarkoordinaten
        self.coordinates = CartToPolar(self.x, self.y)
        polar_coordinates = self.coordinates.calc()

        self.r = polar_coordinates.get("r")

        # s ist als der Abstand der Drehpunkte von Joint J1 und Joint J3 definiert
        self.s = math.sqrt(math.pow(self.r - 9.55 - 1, 2) + math.pow(self.h, 2))

    # berechnet welche Werte die Joints annehmen müssen, um die vorgegebene Zielkoordinate zu erreichen
    def calc_joint_values(self):
        polar_coordinates = self.coordinates.calc()
        psi = polar_coordinates.get("psi")

        # Bereich 1
        if self.r < 26.27:
            l = 26.27 - self.r
            _z = math.sqrt(math.pow(self.l3, 2) - math.pow(l, 2))
            h_j3 = self.h + _z - self.l3
            self.s = math.sqrt(math.pow(15.72, 2) + math.pow(h_j3, 2))
            self.calc_angles()
            alpha_ = math.asin(h_j3 / self.s)
            gamma = math.acos(_z / self.l3)
            j1 = self.alpha + alpha_
            j2 = -math.pi + self.beta
            j3 = -(math.pi / 2) - gamma - j1 - j2

        # Bereich 2
        elif 26.27 <= self.r <= 41.45:
            _z = self.l3
            alpha_ = math.asin(self.h / self.s)
            self.calc_angles()
            j1 = self.alpha + alpha_
            j2 = -math.pi + self.beta
            j3 = -(math.pi / 2) - j1 - j2

        # Bereich 3
        elif self.r > 41.45:
            l = self.r - 41.45
            _z = math.sqrt(math.pow(self.l3, 2) - math.pow(l, 2))
            h_j3 = self.h + _z - self.l3
            self.s = math.sqrt(math.pow(30.9, 2) + math.pow(h_j3, 2))# + 1
            self.calc_angles()
            alpha_ = math.asin(h_j3 / self.s)
            gamma = math.acos(_z / self.l3)
            j1 = self.alpha + alpha_
            j2 = -math.pi + self.beta
            j3 = -(math.pi / 2) + gamma - j1 - j2

        else:
            print("Objekt liegt außerhalb der Reichweite")
            return 1

        # umformen vom Radiant in Grad
        j1_degrees = math.degrees(j1)
        j2_degrees = math.degrees(j2)
        j3_degrees = math.degrees(j3)

        return {"j0": psi + 90, "j1": j1_degrees, "j2": j2_degrees, "j3": j3_degrees, "_z": _z}

    def calc_angles(self):
        # Berechnet die Winkel alpha und beta mit Hilfe des allgemeinen Kosinussatzes
        self.alpha = math.acos((math.pow(self.s, 2) - math.pow(self.l2, 2) + math.pow(self.l1, 2)) / (2 * self.l1 * self.s))
        self.beta = math.acos((-math.pow(self.s, 2) + math.pow(self.l2, 2) + math.pow(self.l1, 2)) / (2 * self.l1 * self.l2))

    def calc_y_range(self):
        '''Berechnet in welchen Bereich der Wert für y liegen sollten, nachdem der Wert für x eingegeben wurde,
        damit der Roboterarm diesen noch erreichen kann, '''
        if 9.54 > self.x > -9.54:
            y_min = math.sqrt(math.pow(9.54, 2) - math.pow(self.x, 2))
        else:
            y_min = -3.4

        y_max = math.sqrt(math.pow(58.18, 2) - math.pow(self.x, 2))

        if y_max < 3.4:
            y_min = -y_max
        return {"y_min": y_min, "y_max": y_max}

    def calc_z_range(self, _z):
        '''Berechnet in welchen Bereich der Wert für z liegen sollten, nachdem die Werte für x und y eingegeben wurden,
        damit der Roboterarm diesen noch erreichen kann'''

        if 20 < self.r < 26.27:
            z_max = 52.4 - _z
            z_max = int(z_max)
            for x in range(z_max):
                self.h = x + self.l3 - 20.6
                z_test = self.calc_joint_values()
                j3 = z_test.get("j3")
                if j3 < -130:
                    z_max = x - 1
                    break

        elif self.r <= 20:
            z_max = 5.1

        elif 26.27 <= self.r <= 41.45:
            z_max = math.sqrt(math.pow(35.56, 2) - math.pow(self.s, 2)) + 3.6

        elif self.r > 41.45:
            z_max = 38.2 - _z

        z_min = 0

        return {"z_min": z_min, "z_max": z_max}


class Input:
    def input_values(self):
        print("Zulässiger Radius: 9,54 cm bis 58,18 cm")
        # Wert für x so definieren, dass die nächste Schleife ausgeführt wird
        x_value = 70

        while x_value < -58.18 or x_value > 58.18:
            print("Zulässige x Werte: -58,18 cm bis 58,18 cm")
            try:
                x_value = eval(input("x Wert: "))
            except SyntaxError:
                continue
            except NameError:
                continue

        y_object = JointValue(x_value, 0, 0)
        y_range = y_object.calc_y_range()
        y_min = y_range.get("y_min")
        y_max = y_range.get("y_max")
        if y_min != 0:
            y_min = y_min + 0.1
            y_max = y_max - 0.1
        # Wert für y so definieren, dass die nächste Schleife ausgeführt wird
        y_value = y_min - 1

        while y_value < y_min or y_value > y_max:
            print("Zulässige y Werte:", y_min, "cm bis", y_max, "cm")
            try:
                y_value = eval(input("y Wert: "))
            except SyntaxError:
                continue
            except NameError:
                continue

        z_object = JointValue(x_value, y_value, 3.6)
        z_joints = z_object.calc_joint_values()
        z_z = z_joints.get("_z")
        z_range = z_object.calc_z_range(z_z)
        z_min = z_range.get("z_min")
        z_max = z_range.get("z_max") - 0.1
        # Wert für z so definieren, dass die nächste Schleife ausgeführt wird
        z_value = z_min - 1

        while z_value < z_min or z_value > z_max:
            print("Zulässige z Werte:", z_min, "cm bis", z_max, "cm")
            try:
                z_value = eval(input("z Wert: "))
            except SyntaxError:
                continue
            except NameError:
                continue

        return {"x": x_value, "y": y_value, "z": z_value}


class Test:
    def __init__(self,):
        print("Zu überprüfender x-Bereich:")
        x_start = eval(input("Start x: "))
        x_end = eval(input("Ende x: "))

        print("Zu überprüfender y-Bereich:")
        y_start = eval(input("Start y: "))
        y_end = eval(input("Ende y: "))

        print("Zu überprüfender z-Bereich:")
        z_start = eval(input("Start z: "))
        z_end = eval(input("Ende z: "))

        i = eval(input("Inkrement: "))

        total = (numpy.abs(x_start) + numpy.abs(x_end)) / i
        progress = 0
        errors = []

        for x in numpy.arange(x_start, x_end, i):
            progress = progress + 100 / total
            print("Fortschritt: ", int(progress), "%")
            if -58.18 < x < 58.18:
                for y in numpy.arange(y_start, y_end, i):
                    y_object = JointValue(x, 0, 0)
                    y_range = y_object.calc_y_range()
                    y_min = y_range.get("y_min")
                    y_max = y_range.get("y_max")
                    if y_min != 0:
                        y_min = y_min + 0.1
                        y_max = y_max - 0.1

                    if y_min < y < y_max:
                        for z in numpy.arange(z_start, z_end, i):
                            z_object = JointValue(x, y, 3.6)
                            z_joints = z_object.calc_joint_values()
                            z_z = z_joints.get("_z")
                            z_range = z_object.calc_z_range(z_z)
                            z_min = z_range.get("z_min")
                            z_max = z_range.get("z_max") - 0.1

                            if z_min < z < z_max:
                                joint_value = JointValue(x, y, z)

                                try:
                                    joint_value.calc_joint_values()

                                except ValueError:
                                    errors.append([x, y, z])

                            else:
                                continue
                    else:
                        continue
            else:
                continue

        number_errors = len(errors)
        if number_errors == 0:
            print("Keine Fehler gefunden.")
        else:
            print("Fehler gefunden bei:")
            for x in range(number_errors):
                print(errors[x])


if __name__ == "__main__":
    proceed = True
    while proceed:
        i = Input()
        output = i.input_values()
        x = output.get("x")
        y = output.get("y")
        z = output.get("z")

        joint_value = JointValue(x, y, z)
        print(joint_value.calc_joint_values())
        while proceed:
            _input = input("Fortsetzen? (y/n): ")

            if _input == "n":
                proceed = False

            elif _input == "t":
                test = Test()

            elif _input == "y":
                break
