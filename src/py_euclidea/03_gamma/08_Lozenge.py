from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A,B,_ = env.add_free_segment(
        (217.0, 290.0), (352.5, 290.5))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A, B)

def construct_goals(A, B):
    result = []
    n_base = vector_perp_rot(A.a - B.a)
    for X,Y in (A,B), (B,A):
        X = X.a
        Y = Y.a
        for n in -n_base, n_base:
            v = Y-X
            d = (v+n)/np.sqrt(2)
            result.append((
                Segment(X, X+d),
                Segment(X+d, Y+d),
                Segment(Y+d, Y),
            ))
    return result

def is_goal_satisfied(goal_ob, goal):
    for g in goal:
        if same_line(g, goal_ob):
            return True
    return False

def get_construction(env, obj):
    # _pg means potential goal
    goal = env.goals[env.goal_index]
    # more solution swap A,B
    a = obj[0]
    b = obj[1]
    for A, B in [[a, b], [b, a]]:
        seg = obj[2]
        AB_len = np.linalg.norm(A.a - B.a)
        p_on_AB = Point((A.a + B.a)/2)
        perp = perp_tool(seg, A)
        p_on_perpendicular_1 = Point(A.a + perp.v * AB_len)
        p_on_perpendicular_2 = Point(A.a - perp.v * AB_len)
        for p_on_perpendicular in [p_on_perpendicular_1, p_on_perpendicular_2]:
            angle_bi_pg = angle_bisector_tool(p_on_perpendicular, A, B)

            # check weather we choose right goal
            if not is_goal_satisfied(angle_bi_pg, goal):
                continue

            circle = circle_tool(A, B)
            multiple_res = intersection_tool(circle, angle_bi_pg)
            # 2 point and both can lead to successful solution
            for p in multiple_res:
                perp_side_pg = perp_tool(perp, p)

                # check weather we choose right goal
                if not is_goal_satisfied(perp_side_pg, goal):
                    continue

                for C in intersection_tool(circle, perp):
                    last_side_pg = line_tool(B, C)
                    if is_goal_satisfied(last_side_pg, goal):
                        return [
                            construction.ConstructionProcess('Perpendicular', [p_on_AB, A]),
                            construction.ConstructionProcess('Angle_Bisector', [p_on_perpendicular, A, B]),
                            construction.ConstructionProcess('Angle_Bisector', [p_on_perpendicular, A, B]),
                            construction.ConstructionProcess('Circle', [A, B]),
                            construction.ConstructionProcess('Perpendicular', [p_on_perpendicular, p]),
                            construction.ConstructionProcess('Perpendicular', [p_on_perpendicular, p]),
                            construction.ConstructionProcess('Line', [C, B]),
                            construction.ConstructionProcess('Line', [C, B]),
                        ], [
                            perp,
                            p_on_perpendicular,
                            angle_bi_pg,
                            circle,
                            p,
                            perp_side_pg,
                            C,
                            last_side_pg
                        ]
    raise Exception("unable to build construction")
