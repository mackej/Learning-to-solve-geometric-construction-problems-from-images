from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A,B,_ = env.add_free_segment(
        (268.0, 324.5), (397.5, 324.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A, B)

def construct_goals(A_in, B_in):
    result = []
    for (A, B) in (A_in, B_in), (B_in, A_in):
        a60 = np.pi/3
        O = rotate_about_point(A, B, a60)
        vertices = [
            rotate_about_point(A, O, a60*i)
            for i in range(1,5)
        ]
        result.append(tuple(
            segment_tool(X,Y)
            for X,Y in zip([A]+vertices, vertices+[B])
        ))
    return result

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A, B = [obj[i] for i in env.goal_par_indices]
    c1 = circle_tool(A, B)
    c2 = circle_tool(B, A)
    P_c, hexa_center = intersection_tool(c1, c2)
    for P_c, hexa_center in (P_c, hexa_center), (hexa_center, P_c):

        l1 = line_tool(P_c, A)
        # it is enough to check only one side, and we can do it early to cut other construction sooner
        if not is_this_goal_satisfied(l1, goal):
            continue
        l2 = line_tool(P_c, B)
        hexa_axis = parallel_tool(line_tool(A, B), hexa_center)
        X = intersection_tool(hexa_axis, l1)
        Y = intersection_tool(hexa_axis, l2)
        AX_half = Point((X.a + A.a) / 2)
        BY_half = Point((Y.a + B.a) / 2)
        l3 = parallel_tool(l1, Y)
        l4 = parallel_tool(l2, X)
        V = intersection_tool(l3, l4)
        l5 = perp_bisector_tool(hexa_center, V)

        return [
                   construction.ConstructionProcess('Circle', [A, B]),
                   construction.ConstructionProcess('Circle', [B, A]),
                   construction.ConstructionProcess('Line', [P_c, A]),
                   construction.ConstructionProcess('Line', [P_c, A]),
                   construction.ConstructionProcess('Line', [P_c, B]),
                   construction.ConstructionProcess('Parallel', [AX_half, Y]),
                   construction.ConstructionProcess('Parallel', [AX_half, Y]),
                   construction.ConstructionProcess('Parallel', [BY_half, X]),
                   construction.ConstructionProcess('Parallel', [BY_half, X]),
                   construction.ConstructionProcess('Perpendicular_Bisector', [V, hexa_center]),
                   construction.ConstructionProcess('Perpendicular_Bisector', [V, hexa_center]),
                   construction.ConstructionProcess('Perpendicular_Bisector', [V, hexa_center]),

               ], [
                   c1, c2,
                   P_c,
                   l1, l2,
                   AX_half, BY_half,
                   X, Y,
                   l3, l4,
                   V,
                   l5
               ]
    raise Exception("cannot get construction of this level")

