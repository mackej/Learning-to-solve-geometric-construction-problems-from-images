from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(360.0, 200.5)
    l = env.add_free_line((8.0, 285.5), (615.5, 289.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(X, l)

def construct_goals(X, l):
    normals = (rotate_vector(l.n, ang) for ang in (-np.pi/3, np.pi/3))
    return tuple(
        (Line(n, np.dot(X.a, n)),)
        for n in normals
    )

def additional_bb(X, l, goal):
    return intersection_tool(l, goal)

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    X, l = [obj[i] for i in env.goal_par_indices]
    pt_on_line_perp = intersection_tool(perp_tool(l, X),l)
    pt_for_rnd_circle = Point(pt_on_line_perp.a + l.v * np.linalg.norm(X.a - pt_on_line_perp.a))

    c1 = circle_tool(X, pt_for_rnd_circle)
    c2 = circle_tool(pt_for_rnd_circle, X)

    p_1 = intersection_tool(c1, l)
    p_2 = intersection_tool(c1, c2)
    for i in p_1:
        for j in p_2:
            res = perp_bisector_tool(i, j)
            if same_line(goal[0], res):
                return [
                    construction.ConstructionProcess('Circle', [X, pt_for_rnd_circle]),
                    construction.ConstructionProcess('Circle', [X, pt_for_rnd_circle]),
                    construction.ConstructionProcess('Circle', [pt_for_rnd_circle, X]),
                    construction.ConstructionProcess('Perpendicular_Bisector', [i, j]),
                    construction.ConstructionProcess('Perpendicular_Bisector', [i, j]),
                    construction.ConstructionProcess('Perpendicular_Bisector', [i, j]),
                    ], [
                    pt_for_rnd_circle,
                    c1,
                    c2,
                    i,
                    j,
                    res,
                    ]
