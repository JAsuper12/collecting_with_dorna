import math


class CartToPolar:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calc(self):
        if self.x > 0:
            psi = math.atan(self.y / self.x)
            psi_degree = math.degrees(psi)

        elif self.x < 0:
            phi = math.atan(self.y / -self.x)
            phi_degree = math.degrees(phi)
            psi_degree = 180 - phi_degree

        elif self.x == 0:
            psi_degree = 90
            r = self.y

        r = math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
        return {"psi": psi_degree, "r": r}


if __name__ == "__main__":
    x_value = eval(input("x Wert: "))
    y_value = eval(input("y Wert: "))
    polar = CartToPolar(x_value, y_value)
    print(polar.calc())
