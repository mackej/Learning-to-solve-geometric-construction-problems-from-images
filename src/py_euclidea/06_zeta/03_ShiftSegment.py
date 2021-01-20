from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A = env.add_free((184.5, 311.5), hidden=False)
    D = env.add_free((281.5, 259.5), hidden=True)
    s = env.add_segment(A, D, hidden=True)

    B = env.add_dep((220.0, 290.0), s)
    env.add_segment(A, B)
    # ray makes C not on |AB|
    l = env.add_ray(B, D, hidden=True)
    #l = env.add_line(A,B, hidden = True)
    C = env.add_dep((419.5, 185.5), l)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
    )
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    v_base = B.a-A.a
    result = []
    for v in v_base, -v_base:
        result.append((Point(C.a+v),))
    return result

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A_in, B_in, C = [obj[i] for i in env.goal_par_indices]
    for A,B in (A_in, B_in), (B_in, A_in):
        l = line_tool(A, B)
        p = perp_bisector_tool(B, C)
        S = intersection_tool(p, l)
        c = circle_tool(S, A)
        R1, R2 = intersection_tool(c, l)
        R = R1
        if same_point(R1, A):
            R = R2
        if same_point(R,goal[0]):
            return [
               construction.ConstructionProcess('Line', [A, B]),
               construction.ConstructionProcess('Perpendicular_Bisector', [B, C]),
               construction.ConstructionProcess('Circle', [S, A]),
               construction.ConstructionProcess('Circle', [S, A]),
               construction.ConstructionProcess('Point', [R]),
           ], [
               l,
               p,
               S,
               c, R
           ]
    raise Exception("cannot get construction of this level")
def additional_degeneration(A,B, C, goal):
    return segment_tool(A, B).contains(goal.a)
