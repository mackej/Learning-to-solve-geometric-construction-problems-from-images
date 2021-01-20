from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(311.5, 371.0)
    A = env.add_free(278.5, 223.0, hidden = True)
    B = env.add_free(355.0, 263.0, hidden = True)
    C = env.add_free(404.5, 230.5, hidden = True)
    la = env.add_line(A, B)
    lb = env.add_line(B, C)
    lc = env.add_constr(parallel_tool, (la, C), Line)
    ld = env.add_constr(parallel_tool, (lb, A), Line)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A, B, C, la, lb, lc, ld, X)

def construct_goals(A, B, C, la, lb, lc, ld, X):
    result = []
    D = intersection_tool(lc, ld)
    for (p1, p2) in (B,D), (A,C):
        dir_line = line_tool(p1, p2)
        result.append((parallel_tool(dir_line, X),))
    return result

def additional_degeneration(A, B, C, la, lb, lc, ld, X, goal):
    cpx1 = complex(*la.v)
    cpx2 = complex(*lb.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
    angle = min(angle, 180-angle)
    return angle < 20
'''
def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A, B, C, la, lb, lc, ld, X = [obj[i] for i in env.goal_par_indices]
    D = intersection_tool(lc, ld)
    for (p1, p2, p3, p4) in (B, D, A, C), (A, C, B, D):
        c1 = circle_tool(p1, X)
        c2 = circle_tool(p2, X)
        Y, Y_d = intersection_tool(c1, c2)
        if same_point(X, Y):
            Y = Y_d
        l = line_tool(p1, Y)
        Z, Z_d = intersection_tool(l, c1)
        if same_point(Z, Y):
            Z = Z_d
        g = line_tool(X, Z)
        if is_this_goal_satisfied(g, goal):
            return [
               construction.ConstructionProcess('Circle', [p1, X]),
               construction.ConstructionProcess('Circle', [p1, X]),
               construction.ConstructionProcess('Circle', [p2, X]),
               construction.ConstructionProcess('Circle', [p2, X]),
               construction.ConstructionProcess('Line', [p1, Y]),
               construction.ConstructionProcess('Line', [p1, Y]),
               construction.ConstructionProcess('Line', [X, Z]),
               construction.ConstructionProcess('Line', [X, Z]),

                ], [
                    c1,
                    c2,
                    Y,
                    l, Z
                ]
'''
def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A, B, C, la, lb, lc, ld, X = [obj[i] for i in env.goal_par_indices]
    D = intersection_tool(lc, ld)
    for (p1, p2, p3, p4) in (B, D, A, C), (A, C, B, D):
        l = line_tool(p1, p2)
        g = parallel_tool(l, X)
        if is_this_goal_satisfied(g, goal):
            return [
               construction.ConstructionProcess('Line', [p1, p2]),
               construction.ConstructionProcess('Line', [p1, p2]),
               construction.ConstructionProcess('Line', [p1, p2]),
               construction.ConstructionProcess('Parallel', [Point(p1.a + (p1.a - p2.a) / 2), X]),
               #construction.ConstructionProcess('Line', [p3, p4]),
               #construction.ConstructionProcess('Line', [p3, p4]),
               #construction.ConstructionProcess('Parallel', [Point(p3.a + (p3.a - p4.a) / 4), X]),
                ], [
                    l,
                    g,

                ]
    raise Exception("cannot get construction of this level")
