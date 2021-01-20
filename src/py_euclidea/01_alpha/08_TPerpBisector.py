from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A,B,_ = env.add_free_segment((247.4, 242.5), (405.5, 241))
    env.set_tools("Perpendicular_Bisector")
    env.goal_params(A, B)

def construct_goals(A, B):
    return (
        perp_bisector_tool(A, B),
    )

def get_construction(env, obj):
    A = obj[0]
    B = obj[1]
    return [
        construction.ConstructionProcess('Perpendicular_Bisector', [A, B])
    ], [
        perp_bisector_tool(A, B)
    ]
