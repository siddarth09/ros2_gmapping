from typing import List, Tuple

class IntPoint:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class GridLineTraversalLine:
    def __init__(self):
        self.num_points = 0
        self.points: List[IntPoint] = []

class GridLineTraversal:
    @staticmethod
    def grid_line_core(start: IntPoint, end: IntPoint, line: GridLineTraversalLine):
        dx, dy = abs(end.x - start.x), abs(end.y - start.y)
        cnt = 0

        if dy <= dx:  # Horizontal line traversal
            d = 2 * dy - dx
            incr1 = 2 * dy
            incr2 = 2 * (dy - dx)
            x, y = (end.x, end.y) if start.x > end.x else (start.x, start.y)
            xend = start.x if start.x > end.x else end.x
            ydirflag = -1 if start.x > end.x else 1

            line.points.append(IntPoint(x, y))
            cnt += 1

            while x < xend:
                x += 1
                if d < 0:
                    d += incr1
                else:
                    y += ydirflag
                    d += incr2
                line.points.append(IntPoint(x, y))
                cnt += 1
        else:  # Vertical line traversal
            d = 2 * dx - dy
            incr1 = 2 * dx
            incr2 = 2 * (dx - dy)
            y, x = (end.y, end.x) if start.y > end.y else (start.y, start.x)
            yend = start.y if start.y > end.y else end.y
            xdirflag = -1 if start.y > end.y else 1

            line.points.append(IntPoint(x, y))
            cnt += 1

            while y < yend:
                y += 1
                if d < 0:
                    d += incr1
                else:
                    x += xdirflag
                    d += incr2
                line.points.append(IntPoint(x, y))
                cnt += 1

        line.num_points = cnt

    @staticmethod
    def grid_line(start: IntPoint, end: IntPoint, line: GridLineTraversalLine):
        GridLineTraversal.grid_line_core(start, end, line)
        
        if start.x != line.points[0].x or start.y != line.points[0].y:
            line.points.reverse()
