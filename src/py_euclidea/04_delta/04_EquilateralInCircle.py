from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import itertools

def init(env):
    C = env.add_free(330.7, 253.0, hidden = True)
    A = env.add_free(443.5, 253.5)
    env.add_circle(C, A)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(C, A)

def construct_goals(C, A):
    C = C.a
    A = A.a
    vertices = [
        C + rotate_vector(A - C, ang)
        for ang in (0, -2*np.pi/3, 2*np.pi/3)
    ]
    return [
        Segment(X, Y)
        for (X, Y) in itertools.combinations(vertices, 2)
    ]

def get_construction(env, obj):

    C, A = [obj[i] for i in env.goal_par_indices]
    input_circle = obj[2]
    rnd_pt = Point(C.a + rotate_vector(A.a - C.a, np.pi / 4))
    rnd_pt_other = Point(C.a + rotate_vector(A.a - C.a, -np.pi / 4))

    c1 = circle_tool(rnd_pt, A)
    c2 = circle_tool(A, rnd_pt)
    P1, P2 = intersection_tool(c1, c2)

    perp1 = perp_bisector_tool(rnd_pt_other, P1)
    perp2 = perp_bisector_tool(rnd_pt_other, P2)

    X, X_d = intersection_tool(input_circle, perp1)
    if same_point(X, A):
        X = X_d
    Y, Y_d = intersection_tool(input_circle, perp2)
    if same_point(Y, A):
        Y = Y_d

    return [
               construction.ConstructionProcess('Circle', [rnd_pt, A]),
               construction.ConstructionProcess('Circle', [rnd_pt, A]),
               construction.ConstructionProcess('Circle', [A, rnd_pt]),
               construction.ConstructionProcess('Perpendicular_Bisector', [rnd_pt_other, P1]),
               construction.ConstructionProcess('Perpendicular_Bisector', [rnd_pt_other, P1]),
               construction.ConstructionProcess('Perpendicular_Bisector', [rnd_pt_other, P1]),
               construction.ConstructionProcess('Perpendicular_Bisector', [rnd_pt_other, P2]),
               construction.ConstructionProcess('Perpendicular_Bisector', [rnd_pt_other, P2]),
               construction.ConstructionProcess('Line', [X, Y]),
               construction.ConstructionProcess('Line', [X, Y]),
               construction.ConstructionProcess('Line', [X, Y]),
           ], [
               rnd_pt,
               c1,
               c2,
               rnd_pt_other,
               P1,
               P2,
               perp1,
               perp2,
               X,
               Y,
               line_tool(X, Y)
           ]

