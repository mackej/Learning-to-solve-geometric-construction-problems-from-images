from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def seg_init():
    X = random_point()
    Y = random_point()
    v = vector_perp_rot(X-Y)
    return X+v, Y+v

def init(env):
    A = env.add_free(177.5, 249.0, rand_init = False)
    B = env.add_free(242.5, 165.5, rand_init = False)
    env.add_segment(A,B)
    env.add_rand_init((A,B), seg_init)
    C = env.add_free(447.0, 268.5)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
    )
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    return compass_tool(A,B,C)

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]

    AB_half = Point((A.a + B.a)/2)
    AC_half = Point((A.a + C.a) / 2)


    ab = line_tool(A, B)
    ac = line_tool(A, C)
    p1 = parallel_tool(ab, C)
    p2 = parallel_tool(ac, B)
    R = intersection_tool(p1, p2)

    return [
               construction.ConstructionProcess('Line', [A, C]),
               construction.ConstructionProcess('Parallel', [AB_half, C]),
               construction.ConstructionProcess('Parallel', [AC_half, B]),
               construction.ConstructionProcess('Circle', [C, R]),
               construction.ConstructionProcess('Circle', [C, R]),
           ], [
               ac,
               p1,
               p2,
               R,
               circle_tool(C, R)
           ]

