from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(263.5, 313.0)
    Y = env.add_free(404.5, 181.0)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(X, Y)

def construct_goals(X, Y):
    X = X.a
    Y = Y.a
    v = vector_perp_rot(X-Y)/2
    A = X + v
    B = Y + v
    C = Y - v
    D = X - v
    return (
        Segment(A, B),
        Segment(B, C),
        Segment(C, D),
        Segment(D, A),
    )

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    X, Y = [obj[i] for i in env.goal_par_indices]
    l_1 = line_tool(X, Y)
    c_1 = circle_tool(X, Y)
    perp_1 = perp_tool(l_1, X)
    P1, P2 = intersection_tool(perp_1, c_1)
    return [
               construction.ConstructionProcess('Line', [X, Y]),
               construction.ConstructionProcess('Circle', [X, Y]),
               construction.ConstructionProcess('Perpendicular', [Point((X.a+Y.a)/2), X]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P1, X]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P1, X]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P2, X]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P2, X]),
               construction.ConstructionProcess('Perpendicular', [Point((X.a+Y.a)/2), Y]),
           ], [
               l_1,
               c_1,
               perp_1,
               P1,
               P2,
               perp_bisector_tool(P1, X),
               perp_bisector_tool(P2, X),
               perp_tool(l_1, Y)
           ]
