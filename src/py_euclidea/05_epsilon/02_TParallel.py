from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    X = env.add_free(303.5, 210.0)
    l = env.add_free_line((2.0, 298.0), (630.5, 306.5))

    env.set_tools("Parallel")
    env.goal_params(l, X)

def construct_goals(l, X):
    return parallel_tool(l, X)

def additional_bb(l, X, goal):
    return Point(l.closest_on(X.a))

def get_construction(env, obj):
    l, X = [obj[i] for i in env.goal_par_indices]
    perp1 = perp_tool(l, X)
    p_on_l = intersection_tool(perp1, l)
    return [
               construction.ConstructionProcess('Parallel', [p_on_l, X]),
           ], [
               parallel_tool(l, X)
           ]
