from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    (A,B,C),_ = env.add_free_triangle(
        (155.5, 360.5), (504.5, 360.5), (395.5, 149.5))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    Ma = midpoint_tool(B,C)
    Mb = midpoint_tool(C,A)
    Mc = midpoint_tool(A,B)
    return circumcircle_tool(Ma,Mb,Mc)


def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]
    # for this construction its best to choose shortest of 3 sides, so we don't have to extend triangle sides.
    min = float("inf")
    bestA, bestB, bestC = None, None, None
    for A_c, B_c, C_c in ((A, B, C), (A, C, B), (B, C, A)):
        q_dist = np.sum((A_c.a -B_c.a)**2)
        if q_dist <= min:
            min = q_dist
            bestA, bestB, bestC = A_c, B_c, C_c
    A, B, C = bestA, bestB, bestC
    p1 = perp_bisector_tool(A, B)
    AB_half = Point((A.a + B.a)/2)
    c1 = circle_tool(AB_half, A)
    bc = line_tool(B, C)
    ac = line_tool(A, C)
    X, X2 = intersection_tool(bc, c1)
    if same_point(X, B):
        X = X2
    Y, Y2 = intersection_tool(ac, c1)
    if same_point(Y, A):
        Y = Y2
    p2 = perp_bisector_tool(X, Y)
    p3 = perp_bisector_tool(X, AB_half)
    S = intersection_tool(p2, p3)
    g = circle_tool(S, X)
    return [
               construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
               construction.ConstructionProcess('Circle', [AB_half, A]),
               construction.ConstructionProcess('Circle', [AB_half, A]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, AB_half]),
               construction.ConstructionProcess('Circle', [S, X]),
               construction.ConstructionProcess('Circle', [S, X]),
           ], [
               p1, AB_half,
               c1,
               X, Y,
               p2,
               p3,
               S, g
           ]

