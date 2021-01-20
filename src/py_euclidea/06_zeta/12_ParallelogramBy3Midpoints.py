from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(245.5, 300.0)
    Y = env.add_free(367.0, 210.5)
    Z = env.add_free(405.5, 284.0)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(X,Y,Z)

def construct_goals(X_in, Y_in, Z_in):
    result = []
    for (X,Y,Z) in (X_in,Y_in,Z_in), (Y_in,Z_in,X_in), (Z_in,X_in,Y_in):
        v = Z.a - (X.a+Y.a)/2
        A = Point(X.a+v)
        B = Point(Y.a+v)
        C = Point(Y.a-v)
        D = Point(X.a-v)
        result.append((
            segment_tool(A,B),
            segment_tool(B,C),
            segment_tool(C,D),
            segment_tool(D,A),
        ))
    return result

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    X_in, Y_in, Z_in = [obj[i] for i in env.goal_par_indices]
    (A, B, C) =[(X_in,Y_in,Z_in), (Y_in,Z_in,X_in), (Z_in,X_in,Y_in)][env.goal_index]
    c1 = compass_tool(A, C, B)
    c2 = compass_tool(B, C, A)
    P1, P2 = intersection_tool(c1, c2)

    int_res = intersection_tool(segment_tool(P1, C), segment_tool(A, B), exception_on_fail=False)
    if not (int_res is None):
        P1, P2 = P2, P1
    g1 = line_tool(P1, C)
    l1 = line_tool(P2, C)
    g2 = parallel_tool(g1, P2)
    g3 = parallel_tool(l1, A)
    g4 = parallel_tool(l1, B)
    P2C_half = Point((P2.a + C.a) / 2)
    P1C_half = Point((P1.a + C.a) / 2)
    # if is_this_goal_satisfied(g1, goal) and is_this_goal_satisfied(g2, goal) and is_this_goal_satisfied(g3, goal):
    return [
               construction.ConstructionProcess('Compass', [A, C, B]),
               construction.ConstructionProcess('Compass', [B, C, A]),
               construction.ConstructionProcess('Line', [P1, C]),
               construction.ConstructionProcess('Line', [P1, C]),
               construction.ConstructionProcess('Line', [P2, C]),
               construction.ConstructionProcess('Line', [P2, C]),
               construction.ConstructionProcess('Parallel', [P1C_half, P2]),
               construction.ConstructionProcess('Parallel', [P2C_half, A]),
               construction.ConstructionProcess('Parallel', [P2C_half, B]),

           ], [
               c1, c2,
               P1, P2,
               l1, g1,
               g2,
               g3,
               g4
           ]


