from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction


def init(env):
    A = env.add_free(225.5, 264.5)
    B = env.add_free(328.5, 186)
    C = env.add_free(443, 310)
    env.set_tools("Line")
    env.goal_params(A, B, C)


def construct_goals(A, B, C):
    return (
        segment_tool(A, B),
        segment_tool(B, C),
        segment_tool(C, A),
    )


def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    C = obj[2]
    return [
        construction.ConstructionProcess('Line', [A, B]),
        construction.ConstructionProcess('Line', [B, C]),
        construction.ConstructionProcess('Line', [A, C]),
    ], [
        line_tool(A, B),
        line_tool(B, C),
        line_tool(A, C)
    ]

def get_tool_hints():
    return [
        "Line",
        "Line",
        "Line"
    ]
