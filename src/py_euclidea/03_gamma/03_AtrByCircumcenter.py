from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A, ray1, ray2 = env.add_free_angle(
        (224.5, 298.0), (580.5, 301.0), (508.5, 31.5))
    O = env.add_free(378.0, 258.0, rand_init = False)
    env.add_rand_init(O, random_point_in_angle, (ray1, ray2))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, ray1, ray2, O)

def construct_goals(A, ray1, ray2, O):
    endpoints = [
        A.a + 2 * ray.v * np.dot(ray.v, O.a-A.a)
        for ray in (ray1, ray2)
    ]
    return Segment(*endpoints)

def get_construction(env, obj):
    A = obj[0]
    ray_1 = obj[2]
    ray_2 = obj[4]
    center = obj[5]
    circle = circle_tool(center, A)
    B_1, B_2 = intersection_tool(circle, ray_1)
    C_1, C_2 = intersection_tool(circle, ray_2)
    if same_point(A.a, B_1.a):
        B = B_2
    else:
        B = B_1
    if same_point(A.a, C_1.a):
        C = C_2
    else:
        C = C_1

    # sometimes this task very degenerated to B or C does not exists. So we make sure we can construct
    # line(B, C) if not line tool will throw exception and we will generate new hopefully not generated level

    return [

        construction.ConstructionProcess('Circle', [center, A]),
        construction.ConstructionProcess('Line', [B, C]),
        construction.ConstructionProcess('Line', [B, C]),
        construction.ConstructionProcess('Line', [B, C]),
    ], [
        circle,
        B,
        C,
        line_tool(B, C)
    ]
