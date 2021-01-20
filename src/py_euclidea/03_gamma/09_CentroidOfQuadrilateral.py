from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(242.0, 168.5, rand_init = False)
    B = env.add_free(370.0, 188.0, rand_init = False)
    C = env.add_free(482.0, 354.0, rand_init = False)
    D = env.add_free(159.0, 354.0, rand_init = False)
    env.add_rand_init((A,B,C,D), random_convex_quadrilateral)
    for (X,Y) in (A,B),(B,C),(C,D),(D,A):
        env.add_segment(X,Y)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(A,B,C,D)

def construct_goals(A,B,C,D):
    return Point((A.a+B.a+C.a+D.a)/4)

def get_construction(env, obj):
    A,B,C,D = [obj[i] for i in env.goal_par_indices]
    AB_half = Point((A.a + B.a)/2)
    CD_half = Point((C.a + D.a)/2)
    Result = Point((A.a+B.a+C.a+D.a)/4)

    BC_half = Point((B.a + C.a)/2)
    DA_half = Point((A.a + D.a)/2)

    return [

        construction.ConstructionProcess('Perpendicular_Bisector', [A, B]),
        construction.ConstructionProcess('Perpendicular_Bisector', [C, D]),
        construction.ConstructionProcess('Line', [AB_half, CD_half]),
        construction.ConstructionProcess('Line', [AB_half, CD_half]),
        construction.ConstructionProcess('Line', [AB_half, CD_half]),
        construction.ConstructionProcess('Perpendicular_Bisector', [AB_half, CD_half]),
        construction.ConstructionProcess('Point', [Result]),
    ], [
        perp_bisector_tool(A, B),
        perp_bisector_tool(C, D),
        AB_half,
        CD_half,
        line_tool(AB_half, CD_half),
        perp_bisector_tool(AB_half, CD_half),
        Result,

        #perp_bisector_tool(B, C),
        #perp_bisector_tool(A, D),
        #BC_half,
        #DA_half,
        #line_tool(BC_half, DA_half),
        #perp_bisector_tool(AB_half, CD_half),
    ]
