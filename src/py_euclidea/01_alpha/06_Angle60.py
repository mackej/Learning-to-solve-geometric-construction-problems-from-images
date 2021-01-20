from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    _,_,ray = env.add_free_ray((243, 268), (384, 268))
    env.set_tools("move", "Point", "Line", "Circle", "intersection")
    env.goal_params(ray)

def construct_goals(ray):
    vecs = (rotate_vector(ray.v, ang) for ang in (-np.pi/3, np.pi/3))
    X = ray.start_point
    return tuple(
        (Ray(X, v),)
        for v in vecs
    )
def get_construction(env, obj):
    ray = obj[2]
    X = Point(ray.start_point)
    rnd_point_on_ray = Point(X.a + ray.v * 100)
    g = env.cur_goal()
    C0, C1 = intersection_tool(
        circle_tool(X, rnd_point_on_ray),
        circle_tool(rnd_point_on_ray, X),
    )

    result = line_tool(C0, X)
    r2 = line_tool(C1,X)
    tol = 0.001
    if (np.abs(result.n - g[0].n) < tol).all() or (np.abs(result.n - (-1) * g[0].n) < tol).all():
        C_point = C0
    else:
        C_point = C1

    return [
        #construction.ConstructionProcess('point', [rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [rnd_point_on_ray, X]),
        #construction.ConstructionProcess('Point', [C_point]),
        construction.ConstructionProcess('Line', [C_point, X]),
        construction.ConstructionProcess('Line', [C_point, X]),
    ], [
        rnd_point_on_ray,
        circle_tool(X, rnd_point_on_ray),
        circle_tool(rnd_point_on_ray, X),
        C_point,
        line_tool(C_point, X)
    ]

def additional_bb(ray, goal):
    return Point(ray.start_point + ray.v + goal.v)
