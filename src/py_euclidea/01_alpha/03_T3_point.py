from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(298, 200.5, hidden = True)
    B = env.add_free(230, 276.5, hidden = True)
    C = env.add_free(392.5, 290.5, hidden = True)
    env.add_line(A, B)
    env.add_line(B, C)
    env.add_line(C, A)
    env.set_tools("Point", "intersection")
    env.goal_params(A, B, C)

def construct_goals(A, B, C):
    return [Point(X.a) for X in (A, B, C)]

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C = obj[2]
    return [construction.ConstructionProcess('Point', [X]) for X in (A, B, C)], [A, B, C]


def get_tool_hints():
    return [
        "intersection",
    ]
