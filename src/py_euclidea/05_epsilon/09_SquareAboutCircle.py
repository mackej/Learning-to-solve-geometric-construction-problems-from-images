from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def line_init():
    X = random_point()
    Y = random_point()
    v = vector_perp_rot(X-Y)
    return X+v, Y+v

def init(env):
    circ = env.add_free_circ((309.5, 199.0), 73.0, hidden_center = False)
    X = env.add_free((17.0, 365.0), hidden = True, rand_init = False)
    Y = env.add_free((623.5, 365.0), hidden = True, rand_init = False)
    env.add_rand_init((X, Y), line_init)
    dir_line = env.add_line(X, Y)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(circ, dir_line)

def construct_goals(circ, dir_line):
    O = circ.c
    v1 = circ.r * dir_line.n
    v2 = circ.r * dir_line.v
    A = O+v1+v2
    B = O+v1-v2
    C = O-v1-v2
    D = O-v1+v2
    return (
        Segment(A, B),
        Segment(B, C),
        Segment(C, D),
        Segment(D, A),
    )

def additional_bb(circ, dir_line, goal):
    return Point(dir_line.closest_on(circ.c))

def get_construction(env, obj):
    circ, dir_line = [obj[i] for i in env.goal_par_indices]
    center = Point(circ.c)
    ax1 = perp_tool(dir_line, center)
    p_on_dir = intersection_tool(dir_line, ax1)
    ax2 = perp_tool(ax1, center)

    A, C = intersection_tool(circ, ax1)
    B, D = intersection_tool(circ, ax2)

    p_on_ax_1 = Point((A.a + circ.c) / 2)
    p_on_ax_2 = Point((B.a + circ.c) / 2)

    return [
               construction.ConstructionProcess('Perpendicular', [p_on_dir, center]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_1, center]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_1, A]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_1, A]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_1, C]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_1, C]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_2, B]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_2, B]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_2, D]),
               construction.ConstructionProcess('Perpendicular', [p_on_ax_2, D]),
           ], [
            ax1,
            ax2,
            A, B, C, D,
            parallel_tool(ax2, A),
            parallel_tool(ax2, C),
            parallel_tool(ax1, B),
            parallel_tool(ax1, D),
           ]

def additional_degeneration(circ, dir_line, goal):
    len = np.linalg.norm(dir_line.closest_on(circ.c) - circ.c)
    return len < circ.r
