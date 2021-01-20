from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    l1 = env.add_free_line((13.5, 204.0), (622.5, 204.0))
    X = env.add_free(342.0, 328.5, hidden = True)
    l2 = env.add_constr(parallel_tool, (l1, X), Line)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(l1, l2)

def construct_goals(l1, l2):
    if np.dot(l1.n, l2.n) > 0: c2 = l2.c
    else: c2 = -l2.c
    return Line(l1.n, (l1.c+c2)/2)

def bbox_dependant():
    return True

def get_construction(env, obj, corners=None):
    l1, l2 = [obj[i] for i in env.goal_par_indices]
    X = obj[0]
    Y = obj[1]
    if corners is not None:
        X, Y = l1.get_endpoints(corners)
        X = Point(X)
        Y = Point(Y)
    p = perp_bisector_tool(X, Y)
    P1 = intersection_tool(l1, p)
    P2 = intersection_tool(l2, p)
    return [
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [X, Y]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P1, P2]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P1, P2]),
               construction.ConstructionProcess('Perpendicular_Bisector', [P1, P2]),
           ], [
               X, Y,
               p,
               P1, P2,
               perp_bisector_tool(P1, P2)
           ]
