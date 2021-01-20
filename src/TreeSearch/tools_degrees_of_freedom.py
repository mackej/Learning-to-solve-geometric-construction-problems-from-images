import itertools
# Describes if for tool there is important object order. For example for Line there is no difference between
# Line(A,B) and Line(B,C) but for circle first object is always center hence Circle(A,B) != Circle(B,C)
# for Angle_Bisector there is important point the middle one which represent vertex of angle hence
# Angle_Bisector(A,B,C) = Angle_Bisector(C,B,A) != Angle_Bisector(B,C,B)
def get_degrees_of_freedom():
    return {
        "Line": [[0, 1]],
        "Circle": [[0, 1], [1, 0]],
        "Perpendicular_Bisector": [[0, 1]],
        "Angle_Bisector": [[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        "Perpendicular": [[0, 1], [1, 0]],
        "Parallel": [[0, 1], [1, 0]],
        "Compass": [[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        "intersection": [[0, 1]],

    }


