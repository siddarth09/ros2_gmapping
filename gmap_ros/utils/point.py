import math

class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return Point(self.x * value, self.y * value)
        elif isinstance(value, Point):
            return self.x * value.x + self.y * value.y
        else:
            raise TypeError("Unsupported type for multiplication")

    def __rmul__(self, value):
        return self.__mul__(value)

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

class OrientedPoint(Point):
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        super().__init__(x, y)
        self.theta = theta

    def normalize(self):
        if -math.pi <= self.theta < math.pi:
            return
        multiplier = int(self.theta / (2 * math.pi))
        self.theta = self.theta - multiplier * 2 * math.pi
        if self.theta >= math.pi:
            self.theta -= 2 * math.pi
        if self.theta < -math.pi:
            self.theta += 2 * math.pi

    def rotate(self, alpha):
        s = math.sin(alpha)
        c = math.cos(alpha)
        a = alpha + self.theta
        a = math.atan2(math.sin(a), math.cos(a))
        return OrientedPoint(c * self.x - s * self.y, s * self.x + c * self.y, a)

    def __add__(self, other):
        return OrientedPoint(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other):
        return OrientedPoint(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return OrientedPoint(self.x * value, self.y * value, self.theta * value)
        else:
            raise TypeError("Unsupported type for multiplication")

    @staticmethod
    def absolute_difference(p1, p2):
        delta = p1 - p2
        delta.theta = math.atan2(math.sin(delta.theta), math.cos(delta.theta))
        s = math.sin(p2.theta)
        c = math.cos(p2.theta)
        return OrientedPoint(c * delta.x + s * delta.y, -s * delta.x + c * delta.y, delta.theta)

    @staticmethod
    def absolute_sum(p1, p2):
        s = math.sin(p1.theta)
        c = math.cos(p1.theta)
        return OrientedPoint(c * p2.x - s * p2.y, s * p2.x + c * p2.y, p2.theta) + p1

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

# Example usage
if __name__ == "__main__":
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    print("p1 + p2:", p1 + p2)
    print("p1 - p2:", p1 - p2)
    print("p1 * 2:", p1 * 2)
    print("Dot product p1 * p2:", p1 * p2)

    op1 = OrientedPoint(1, 2, math.pi / 4)
    op2 = OrientedPoint(3, 4, -math.pi / 4)
    print("op1 + op2:", op1 + op2)
    print("Absolute difference:", OrientedPoint.absolute_difference(op1, op2))
    print("Euclidean distance between p1 and p2:", Point.euclidean_distance(p1, p2))
