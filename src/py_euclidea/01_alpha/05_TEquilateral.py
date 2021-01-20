from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A, B, _ = env.add_free_segment((274.5, 225.5), (391.5, 226))
    env.set_tools("move", "Point", "Line", "Circle", "intersection")
    env.goal_params(A, B)

def construct_goals(A, B):
    C0, C1 = intersection_tool(
        circle_tool(A, B),
        circle_tool(B, A),
    )
    return (
        (segment_tool(C0, A), segment_tool(C0, B)),
        (segment_tool(C1, A), segment_tool(C1, B)),
    )

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    g = env.cur_goal()
    goal_C_point = Point(g[0].end_points[0])
    return[
        construction.ConstructionProcess('Circle', [A, B]),
        construction.ConstructionProcess('Circle', [B, A]),
        #construction.ConstructionProcess('Point', [goal_C_point]),
        construction.ConstructionProcess('Line', [A, goal_C_point]),
        construction.ConstructionProcess('Line', [A, goal_C_point]),
        construction.ConstructionProcess('Line', [B, goal_C_point])
    ], [
        circle_tool(A, B),
        circle_tool(B, A),
        goal_C_point,
        line_tool(A, goal_C_point),
        line_tool(goal_C_point, B)
    ]
