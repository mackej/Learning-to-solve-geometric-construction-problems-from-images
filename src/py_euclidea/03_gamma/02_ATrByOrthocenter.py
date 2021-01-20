from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A, ray1, ray2 = env.add_free_angle(
        (158.0, 293.5), (578.0, 295.0), (542.5, 24.5))
    H = env.add_free(388.5, 207.0, rand_init = False)
    env.add_rand_init(H, random_point_in_angle, (ray1, ray2))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, ray1, ray2, H)

def construct_goals(A, ray1, ray2, H):
    alt1 = perp_tool(ray1, H)
    alt2 = perp_tool(ray2, H)
    B = intersection_tool(alt1, ray2)
    C = intersection_tool(alt2, ray1)
    return segment_tool(B, C)

def get_construction(env, obj):
    ray_1 = obj[2]
    ray_2 = obj[4]
    ortocenter = obj[5]
    r1_perp = perp_tool(ray_1, ortocenter)
    r2_perp = perp_tool(ray_2, ortocenter)
    ray_P1 = intersection_tool(ray_1, r1_perp)
    ray_P2 = intersection_tool(ray_2, r2_perp)
    triangle_P1 = intersection_tool(r1_perp, ray_2)
    triangle_P2 = intersection_tool(r2_perp, ray_1)

    return [

        construction.ConstructionProcess('Perpendicular', [ray_P1, ortocenter]),
        construction.ConstructionProcess('Perpendicular', [ray_P2, ortocenter]),
        construction.ConstructionProcess('Line', [triangle_P1, triangle_P2]),
        construction.ConstructionProcess('Line', [triangle_P1, triangle_P2]),
        construction.ConstructionProcess('Line', [triangle_P1, triangle_P2]),
    ], [
        r1_perp,
        r2_perp,
        triangle_P1,
        triangle_P2,
        line_tool(triangle_P1, triangle_P2)
    ]
