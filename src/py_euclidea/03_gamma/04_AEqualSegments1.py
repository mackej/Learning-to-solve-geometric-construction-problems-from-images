from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    B, a, c = env.add_free_angle(
        (156.0, 286.5), (546.0, 26.5), (611.5, 286.0))
    M = env.add_free(369.5, 184.0, rand_init = False)
    env.add_rand_init(M, random_point_in_angle, (a, c))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(B, a, c, M)

def construct_goals(B, a, c, M):
    D = intersection_tool(perp_bisector_tool(B, M), a)
    E1, E2 = intersection_tool(circle_tool(M, D), c)
    l2 = segment_tool(D, M)
    return [
        (l2, segment_tool(M, E))
        for E in (E1, E2)
        if E is not None
    ]

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A = obj[0]
    ray_1 = obj[2]
    ray_2 = obj[4]
    M = obj[5]
    perp = perp_bisector_tool(A, M)
    pot_sol_p1 = intersection_tool(perp, ray_1)
    pot_sol_p2 = intersection_tool(perp, ray_2)
    s = segment_tool(pot_sol_p1,M)
    if same_line(goal[0],s) or same_line(goal[1], s):
        pt_on_sol = pot_sol_p1
        other_ray = ray_2
    else:
        pt_on_sol = pot_sol_p2
        other_ray = ray_1
    circle = circle_tool(M, pt_on_sol)
    other_point_1, other_point_2 = intersection_tool(circle, other_ray)
    s2 = segment_tool(other_point_1, M)

    if same_line(goal[0],s2) or same_line(goal[1],s2):
        other_point = other_point_1
    else:
        other_point = other_point_2
    return [

        construction.ConstructionProcess('Perpendicular_Bisector', [M, A]),
        construction.ConstructionProcess('Circle', [M, pt_on_sol]),
        construction.ConstructionProcess('Circle', [M, pt_on_sol]),
        construction.ConstructionProcess('Line', [M, pt_on_sol]),
        construction.ConstructionProcess('Line', [M, other_point]),
        construction.ConstructionProcess('Line', [M, other_point]),
    ], [
        perp,
        pt_on_sol,
        circle,
        other_point,
        line_tool(M, other_point)
    ]
