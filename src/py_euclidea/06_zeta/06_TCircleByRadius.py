from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def seg_init():
    X = random_point()
    Y = random_point()
    v = vector_perp_rot(X-Y)
    return X+v, Y+v

def init(env):
    A = env.add_free(216.0, 249.5, rand_init = False)
    B = env.add_free(280.5, 165.5, rand_init = False)
    env.add_segment(A,B)
    env.add_rand_init((A,B), seg_init)
    C = env.add_free(390.0, 268.5)

    env.set_tools("Compass")
    env.goal_params(A,B,C)

def construct_goals(A,B,C):
    return compass_tool(A,B,C)

def get_construction(env, obj):
    A, B, C = [obj[i] for i in env.goal_par_indices]

    return [
               construction.ConstructionProcess('Compass', [A, B, C]),
           ], [
               compass_tool(A, B, C)
           ]
