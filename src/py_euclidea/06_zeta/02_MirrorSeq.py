from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A,B,_ = env.add_free_segment(
        (246.5, 171.0), (191.0, 301.0))
    l = env.add_free_line(
        (332.0, 7.0), (331.0, 478.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A,B,l)

def construct_goals(A,B,l):
    A2 = reflect_by_line(A,l)
    B2 = reflect_by_line(B,l)
    return (
        A2, B2, segment_tool(A2,B2)
    )

def get_construction(env, obj):
    A, B, l = [obj[i] for i in env.goal_par_indices]
    A2 = reflect_by_line(A, l)
    B2 = reflect_by_line(B, l)
    if np.sum((A.a - A2.a)**2) >= np.sum((B.a - B2.a)**2):
        A, B, A2, B2 = B, A, B2, A2
    rnd_pt_on_l = intersection_tool(l, perp_tool(l, Point((A.a+B.a)/2)))
    p1 = perp_tool(l, A)
    p2 = perp_tool(l, B)
    S_a = intersection_tool(p1, l)
    S_b = intersection_tool(p2, l)
    c1 = circle_tool(S_a, A)
    c2 = circle_tool(S_b, B)

    return [
               construction.ConstructionProcess('Perpendicular', [rnd_pt_on_l, A]),
               construction.ConstructionProcess('Circle', [S_a, A]),
               construction.ConstructionProcess('Circle', [S_a, A]),
               construction.ConstructionProcess('Perpendicular', [rnd_pt_on_l, B]),
               construction.ConstructionProcess('Circle', [S_b, B]),
               construction.ConstructionProcess('Circle', [S_b, B]),
               construction.ConstructionProcess('Line', [A2, B2]),
               construction.ConstructionProcess('Line', [A2, B2]),
               construction.ConstructionProcess('Line', [A2, B2]),
           ], [
               p1, p2,
               S_a, S_b,
               c1, c2,
               A2, B2,
               line_tool(A2, B2)
           ]
