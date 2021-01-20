from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(254.5, 173.5)
    B = env.add_free(376.0, 324.0)
    l = env.add_free_line((27.0, 274.0), (621.5, 269.5))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
    )
    env.goal_params(A, B, l)

def construct_goals(A, B, l):
    A2 = reflect_by_line(A, l)
    X = intersection_tool(line_tool(B, A2), l)
    return (ray_tool(X,A), ray_tool(X,B))

def ini_check(A,B,l, goal, scale):
    ddistA = np.dot(l.n, A.a)
    ddistB = np.dot(l.n, B.a)
    return (ddistA-l.c) * (ddistB-l.c) < 0

def get_construction(env, obj):
    A, B, l = [obj[i] for i in env.goal_par_indices]
    p1 = Point(l.closest_on(A.a))
    c = circle_tool(p1, A)
    X1, X2 = intersection_tool(c, l)
    c2 = circle_tool(X1, A)
    Y, Y2 =intersection_tool(c, c2)
    if same_point(A, Y):
        Y = Y2
    g1 = line_tool(Y, B)
    V = intersection_tool(g1, l)

    return [
               construction.ConstructionProcess('Circle', [p1, A]),
               construction.ConstructionProcess('Circle', [p1, A]),
               construction.ConstructionProcess('Circle', [X1, A]),
               construction.ConstructionProcess('Circle', [X1, A]),
               construction.ConstructionProcess('Line', [Y, B]),
               construction.ConstructionProcess('Line', [Y, B]),
               construction.ConstructionProcess('Line', [V, A]),
               construction.ConstructionProcess('Line', [V, A]),
           ], [
               p1,
               c,
               X1, c2,
               g1, V,
               line_tool(V, A)
           ]

def additional_degeneration(A, B, l, goal):
    x, y = goal[0], goal[1]
    cpx1 = complex(*x.v)
    cpx2 = complex(*y.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
    angle = min(angle, 180 - angle)
    return angle < 20


