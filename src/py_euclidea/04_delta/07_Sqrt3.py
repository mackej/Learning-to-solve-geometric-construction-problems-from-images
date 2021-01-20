from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(238.5, 256.0)
    B = env.add_free(355.0, 255.5)
    env.add_ray(A, B)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, B)

def construct_goals(A, B):
    coef = np.sqrt(3)
    return Point(coef*B.a + (1-coef)*A.a)

def get_construction(env, obj):
    goal = env.goals[env.goal_index][0]
    A, B = [obj[i] for i in env.goal_par_indices]
    l = obj[2]
    c_1 = circle_tool(B, A)
    P = Point(B.a + (B.a - A.a))
    c_2 = circle_tool(P, B)
    X, _ = intersection_tool(c_1, c_2)
    c_3 = circle_tool(A, X)
    return [
               construction.ConstructionProcess('Circle', [B, A]),
               construction.ConstructionProcess('Circle', [P, B]),
               construction.ConstructionProcess('Circle', [P, B]),
               construction.ConstructionProcess('Circle', [A, X]),
               construction.ConstructionProcess('Circle', [A, X]),
               construction.ConstructionProcess('Point', [goal]),
           ], [
               c_1,
               P,
               c_2,
               X,
               c_3,
               goal
           ]


