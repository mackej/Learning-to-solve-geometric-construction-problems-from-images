from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction



def init(env):
    vert1,_ = env.add_free_rectangle(
        (472.5, 365.0), (272.0, 365.0), (271.5, 255.5))
    vert2,_ = env.add_free_rectangle(
        (315.0, 161.0), (216.5, 197.5), (191.0, 129.5))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(*(vert1+vert2))

def construct_goals(A,B,C,D,E,F,G,H):
    X1 = Point(np.average((A.a,B.a,C.a,D.a), axis = 0))
    X2 = Point(np.average((E.a,F.a,G.a,H.a), axis = 0))
    return line_tool(X1, X2)

def get_construction(env, obj):
    A, B, C, D, E, F, G, H = [obj[i] for i in env.goal_par_indices]
    X1 = Point(np.average((A.a, B.a, C.a, D.a), axis=0))
    X2 = Point(np.average((E.a, F.a, G.a, H.a), axis=0))
    return [
               construction.ConstructionProcess('Line', [A, C]),
               construction.ConstructionProcess('Line', [B, D]),
               construction.ConstructionProcess('Line', [E, G]),
               construction.ConstructionProcess('Line', [F, H]),
               construction.ConstructionProcess('Line', [X1, X2]),
               construction.ConstructionProcess('Line', [X1, X2]),
               construction.ConstructionProcess('Line', [X1, X2]),
           ], [
               line_tool(A, C),
               line_tool(B, D),
               line_tool(E, G),
               line_tool(F, H),
               X1,
               X2,
               line_tool(X1, X2)
           ]
