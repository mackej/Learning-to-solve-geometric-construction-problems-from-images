from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A = env.add_free(249.5, 227.0)
    B = env.add_free(348.5, 340.5)
    L = env.add_free(8.0, 340.0, hidden = True)
    l = env.add_line(B, L)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, B, l)

def construct_goals(A, B, l):
    b = perp_bisector_tool(A, B)
    p = perp_tool(l, B)
    C = intersection_tool(p, b)
    return circle_tool(C, B)

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    line = obj[3]

    rnd_point_on_line = Point(B.a + line.v * np.linalg.norm(B.a - A.a) / 2)
    center = intersection_tool(perp_bisector_tool(A, B), perp_tool(line, B))
    return [

        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Perpendicular', [rnd_point_on_line, B]),
        construction.ConstructionProcess('Circle', [center, B]),
        construction.ConstructionProcess('Circle', [center, B]),
    ], [
        perp_bisector_tool(A, B),
        perp_tool(line, B),
        center,
        circle_tool(center, B)
    ]
