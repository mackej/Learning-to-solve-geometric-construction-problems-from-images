from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(231.0, 206.5)
    Y = env.add_free(396.5, 205.0)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(X, Y)

def construct_goals(X, Y):
    X = X.a
    Y = Y.a
    v1 = (Y-X)/2
    result = []
    for v2 in -vector_perp_rot(v1), vector_perp_rot(v1):
        A = X+v1-v2
        B = Y+v1+v2
        C = Y+3*v2-v1
        D = X+v2-v1
        result.append((
            Segment(A, B),
            Segment(B, C),
            Segment(C, D),
            Segment(D, A),
        ))
    return result

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    X, Y = [obj[i] for i in env.goal_par_indices]
    perp_bi_1 = perp_bisector_tool(X, Y)
    c1 = circle_tool(X, Y)
    P1, P2 = intersection_tool(c1, perp_bi_1)
    for P1, P2 in [[P1, P2], [P2, P1]]:
        c2 = circle_tool(P1, P2)
        A, A_d = intersection_tool(c2, c1)
        B, B_d = intersection_tool(c2, perp_bi_1)

        if same_point(A, P2):
            A = A_d

        if same_point(B, P2):
            B = B_d

        side_1 = angle_bisector_tool(P1, A, B)
        side_2 = perp_tool(side_1, X)
        V = intersection_tool(perp_bi_1, side_2)
        side_3 = line_tool(V, Y)
        Z = intersection_tool(side_1, perp_bi_1)
        side_4 = perp_tool(side_1, Z)
        if is_this_goal_satisfied(side_1, goal) and is_this_goal_satisfied(side_2, goal) and is_this_goal_satisfied(side_3, goal) and is_this_goal_satisfied(side_4, goal):
            return [
                       construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
                       construction.ConstructionProcess('Circle', [X, Y]),
                       construction.ConstructionProcess('Circle', [P1, P2]),
                       construction.ConstructionProcess('Circle', [P1, P2]),
                       construction.ConstructionProcess('Circle', [P1, P2]),
                       construction.ConstructionProcess('Angle_Bisector', [P1, A, B]),
                       construction.ConstructionProcess('Angle_Bisector', [P1, A, B]),
                       construction.ConstructionProcess('Angle_Bisector', [P1, A, B]),
                       construction.ConstructionProcess('Perpendicular', [A, X]),
                       construction.ConstructionProcess('Perpendicular', [A, Z]),
                       construction.ConstructionProcess('Perpendicular', [A, Z]),
                       construction.ConstructionProcess('Line', [V, Y]),
                       construction.ConstructionProcess('Line', [V, Y]),
                   ], [
                       perp_bi_1,
                       c1,
                       c2,
                       P1,
                       P2,
                       A, B,
                       side_1,
                       Z, V,
                       side_2, side_3, side_4
                   ]
    raise Exception("cannot get construction of this level")
