from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    (A,B,C,D),_ = env.add_free_square(
        (229.5, 322.0), (408.5, 322.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A, B, C, D)

def construct_goals(A,B,C,D):
    X = midpoint_tool(C,D)
    return circumcircle_tool(X,A,B)

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A, B, C, D = [obj[i] for i in env.goal_par_indices]
    for A, B, C, D in (A, B, C, D), (B, C, D, A), (C, D, A, B), (B, C, D, A):
        p1 = perp_bisector_tool(A, B)
        AB_half = Point((A.a + B.a) / 2)
        p2 = perp_bisector_tool(AB_half, C)
        Center = intersection_tool(p1, p2)
        g = circle_tool(Center, AB_half)
        if is_this_goal_satisfied(g, goal):
            return [
                   construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
                   construction.ConstructionProcess('Perpendicular_Bisector', [AB_half, C]),
                   construction.ConstructionProcess('Perpendicular_Bisector', [AB_half, C]),
                   construction.ConstructionProcess('Circle', [Center, AB_half]),
                   construction.ConstructionProcess('Circle', [Center, AB_half]),

               ], [
                   p1,
                   AB_half,
                   p2,
                   Center,
                   g
               ]
    raise Exception("cannot get construction of this level")


