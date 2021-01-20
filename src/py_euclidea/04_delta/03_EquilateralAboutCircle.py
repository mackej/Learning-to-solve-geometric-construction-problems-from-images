from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import itertools

def init(env):
    C = env.add_free(337.0, 243.0)
    A = env.add_free(434.0, 264.0)
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
    lines = []
    for ang in (0, -2*np.pi/3, 2*np.pi/3):
        v = rotate_vector(A - C, ang)
        X = C + v
        lines.append(Line(v, np.dot(v, X)))
    vertices = [
        intersection_tool(l1, l2)
        for (l1, l2) in itertools.combinations(lines, 2)
    ]
    return [
        segment_tool(X, Y)
        for (X, Y) in itertools.combinations(vertices, 2)
    ]

def get_construction(env, obj):

    C, A = [obj[i] for i in env.goal_par_indices]
    input_circle = obj[2]

    line_1 = line_tool(C, A)
    P1, P1_d = intersection_tool(input_circle, line_1)
    if same_point(P1, A):
        P1 = P1_d

    circle_2 = circle_tool(P1, C)
    T1, T1_d = intersection_tool(circle_2, line_1)

    if same_point(T1, C):
        T1 = T1_d

    V1, V2 = intersection_tool(circle_2, input_circle)

    return [
               construction.ConstructionProcess('Line', [C, A]),
               construction.ConstructionProcess('Circle', [P1, C]),
               construction.ConstructionProcess('Circle', [P1, C]),
               construction.ConstructionProcess('Line', [T1, V1]),
               construction.ConstructionProcess('Line', [T1, V1]),
               construction.ConstructionProcess('Line', [T1, V1]),
               construction.ConstructionProcess('Line', [T1, V2]),
               construction.ConstructionProcess('Line', [T1, V2]),
               construction.ConstructionProcess('Perpendicular', [C, A]),
           ], [
               line_1,
               P1,
               circle_2,
               T1,
               V1,
               V2,
               line_tool(T1, V1),
               line_tool(T1, V2),
               perp_tool(line_1, A)
           ]
