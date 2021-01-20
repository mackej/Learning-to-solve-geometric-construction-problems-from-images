from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    (A,B,C,D),(a,b,c,d) = env.add_free_square(
        (420.0, 151.0), (219.0, 150.0))
    X = env.add_dep((274.0, 150.5), a)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(A,B,C,D,X)

def construct_goals(A,B,C,D,X):
    O = (A.a+B.a+C.a+D.a)/4
    v = X.a - O
    vertices = []
    for _ in range(3):
        v = vector_perp_rot(v)
        vertices.append(Point(O + v))
    return [
        segment_tool(p1,p2)
        for p1,p2 in zip([X]+vertices, vertices+[X])
    ]

def get_construction(env, obj):
    A,B,C,D,X = [obj[i] for i in env.goal_par_indices]
    c1 = circle_tool(B, X)
    p, _ = intersection_tool(c1, line_tool(B, C))
    p_on_b = Point(B.a + (C.a-B.a)*1/4)
    par = perp_tool(line_tool(B, C), p)
    Y = intersection_tool(par, line_tool(A, D))
    p_on_xy = Point((X.a + Y.a) /2)
    xy = line_tool(X, Y)
    yz = perp_tool(xy, Y)
    xv = perp_tool(xy, X)

    Z = intersection_tool(yz, line_tool(C, D))
    V = intersection_tool(xv, line_tool(B, C))

    return [
               construction.ConstructionProcess('Circle', [B, X]),
               construction.ConstructionProcess('Perpendicular', [p_on_b, p]),
               construction.ConstructionProcess('Perpendicular', [p_on_b, p]),
               construction.ConstructionProcess('Line', [X, Y]),
               construction.ConstructionProcess('Line', [X, Y]),
               construction.ConstructionProcess('Perpendicular', [p_on_xy, X]),
               construction.ConstructionProcess('Perpendicular', [p_on_xy, Y]),
               construction.ConstructionProcess('Line', [Z, V]),
               construction.ConstructionProcess('Line', [Z, V]),
               construction.ConstructionProcess('Line', [Z, V]),

           ], [
            c1,
            p,
            Y,
            xy, yz, xv,
            Z,
            V,
            line_tool(Z, V)
           ]
