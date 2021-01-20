from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    circ = env.add_free_circ((329.0, 247.0), 118.6, hidden_center = False)
    X = env.add_free(279.0, 196.5)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector", "Perpendicular",
        "intersection",
    )
    env.goal_params(circ, X)

def construct_goals(circ, X):
    n = circ.c - X.a
    line = Line(n, np.dot(n, X.a))
    A,B = intersection_lc(line, circ)
    return Segment(A, B)

def get_construction(env, obj):
    C_center = obj[0]
    ch_center = obj[3]
    l = line_tool(C_center, ch_center)
    return [
        construction.ConstructionProcess('Line', [C_center, ch_center]),
        construction.ConstructionProcess('Perpendicular', [C_center, ch_center]),
    ], [
        l,
        perp_tool(l, ch_center)
    ]
