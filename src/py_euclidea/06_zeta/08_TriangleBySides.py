from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A,B,s = env.add_free_segment(
        (237.0, 346.5), (433.5, 346.5))

    sep = env.add_constr(perp_bisector_tool, (A, B), Line, hidden=True)
    M = env.add_constr(intersection_tool, (sep, s), Point, hidden=True)
    separ_1 = env.add_constr(perp_bisector_tool, (A, M), Line, hidden=True)
    separ_2 = env.add_constr(perp_bisector_tool, (B, M), Line, hidden=True)

    A2 = env.add_free(218.5, 209.5, rand_init=False)
    env.add_rand_init(A2, random_point_in_subplane, (separ_1, A))
    B2 = env.add_free(284.0, 124.0, rand_init=False)
    env.add_rand_init(B2, random_point_in_subplane, (separ_1, A))
    A3 = env.add_free(491.0, 98.5, rand_init=False)
    env.add_rand_init(A3, random_point_in_subplane, (separ_2, B))
    B3 = env.add_free(491.5, 227.0, rand_init=False)
    env.add_rand_init(B3, random_point_in_subplane, (separ_2, B))

    env.add_segment(A2, B2)
    env.add_segment(A3, B3)
    '''
    A2,B2,_ = env.add_free_segment(
       (218.5, 209.5), (284.0, 124.0))
    A3,B3,_ = env.add_free_segment(
        (491.0, 98.5), (491.5, 227.0))
    '''
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(A,B,A2,B2,A3,B3)

def construct_goals(A,B,A2,B2,A3,B3):
    result = []
    for X,Y in (A,B),(B,A):
        c2 = compass_tool(A2,B2,X)
        c3 = compass_tool(A3,B3,Y)
        for C in intersection_tool(c2, c3):
            result.append((
                segment_tool(X,C),
                segment_tool(Y,C),
            ))
            return result

def ini_check(A,B,A2,B2,A3,B3,goal,scale):
    tri_segments = list(goal)+[segment_tool(A,B)]
    segments = [
        segment_tool(A2,B2),
        segment_tool(A3,B3),
    ]
    if is_intersecting_ll(*segments): return False
    for s1 in tri_segments:
        for s2 in segments:
            if is_intersecting_ll(s1, s2):
                return False
    return True

def get_construction(env, obj):
    A,B,A2,B2,A3,B3 = [obj[i] for i in env.goal_par_indices]
    goal = env.goals[env.goal_index]
    C_g = intersection_tool(goal[0],goal[1])
    for X, Y in (A, B), (B, A):
        c2 = compass_tool(A2, B2, X)
        c3 = compass_tool(A3, B3, Y)
        C = intersection_tool(c2, c3)
        for c_point in C:
            if same_point(c_point, C_g):
                return [
               construction.ConstructionProcess('Compass', [A2, B2, X]),
               construction.ConstructionProcess('Compass', [A3, B3, Y]),
               construction.ConstructionProcess('Line', [c_point, A]),
               construction.ConstructionProcess('Line', [c_point, A]),
               construction.ConstructionProcess('Line', [c_point, B]),
           ], [
               c2,
               c3,
               c_point,
               line_tool(c_point, A),
               line_tool(c_point, B)
           ]
def additional_degeneration(A,B,A2,B2,A3,B3, goal):
    # segments can be too similar otherwise its hard to distinguish between parts.
    len1 = np.linalg.norm(A2.a - B2.a)
    len2 = np.linalg.norm(A3.a - B3.a)
    return abs(len1 - len2) < 1/4 * max(len1, len2)
