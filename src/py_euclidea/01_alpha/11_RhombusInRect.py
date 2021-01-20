from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    (A,B,C,D), _ = env.add_free_rectangle(
        (448, 312.5), (204.5, 312.5), (204.5, 142))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(A, B, C, D)

def construct_goals(A_in, B_in, C_in, D_in):
    M = (A_in.a+C_in.a)/2
    result = []
    for (A,B,C,D) in ((A_in, B_in, C_in, D_in),
                      (D_in, C_in, B_in, A_in)):
        diag = perp_bisector_tool(B, D)
        ab = line_tool(A,B)
        cd = line_tool(C,D)
        X_ab = Point(intersection_ll(ab, diag))
        X_cd = Point(intersection_ll(cd, diag))
        result.append((segment_tool(B, X_cd),
                       segment_tool(D, X_ab)))
    return result

def ini_check(A, B, C, D, goal, scale):
    return B.dist_from(A.a) > B.dist_from(C.a)

def get_construction(env, obj):
    g = env.cur_goal()
    A, B, C, D = [env.objs[i] for i in env.goal_par_indices]

    if not same_point(A.a, g[0].end_points[0]) and not same_point(A.a, g[1].end_points[0]):
        A, B , C, D = [B, C, D, A]


    G1P1 = Point(g[0].end_points[0])
    G1P2 = Point(g[0].end_points[1])
    G2P1 = Point(g[1].end_points[0])
    G2P2 = Point(g[1].end_points[1])
    if same_point(G1P1, A) or same_point(G1P1, B) or same_point(G1P1, C) or same_point(G1P1, D):
        new_p1 = G1P2
    else:
        new_p1 = G1P1

    if same_point(G2P1, A) or same_point(G2P1, B) or same_point(G2P1, C) or same_point(G2P1, D):
        new_p2 = G2P2
    else:
        new_p2 = G2P1

    return [
        construction.ConstructionProcess('Perpendicular_Bisector', [A, C]),
        #construction.ConstructionProcess('Point', [P1]),
        #construction.ConstructionProcess('Point', [P2]),
        construction.ConstructionProcess('Line', [G1P1, G1P2]),
        construction.ConstructionProcess('Line', [G1P1, G1P2]),
        construction.ConstructionProcess('Line', [G2P1, G2P2]),
        construction.ConstructionProcess('Line', [G2P1, G2P2]),
    ], [
        perp_bisector_tool(A, C),
        new_p1,
        new_p2,
        line_tool(G1P1, G1P2),
        line_tool(G2P1, G2P2)
    ]
