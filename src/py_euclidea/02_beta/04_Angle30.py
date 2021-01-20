from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    _,_,ray = env.add_free_ray((230.0, 268.0), (621.5, 268.0))
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(ray)

def construct_goals(ray):
    vecs = (rotate_vector(ray.v, ang) for ang in (-np.pi/6, np.pi/6))
    X = ray.start_point
    return tuple(
        (Ray(X, v),)
        for v in vecs
    )

def additional_bb(ray, goal):
    return Point(ray.start_point + ray.v + goal.v)

def scale_able():
    return False

def get_construction(env, obj):
    r = obj[2]
    X = Point(r.start_point)
    rnd_point_on_ray = Point(X.a + r.v * 200)
    g = env.cur_goal()
    C0, C1 = intersection_tool(
        circle_tool(X, rnd_point_on_ray),
        circle_tool(rnd_point_on_ray, X),
    )

    angle1 = angle_bisector_tool(C0, X, rnd_point_on_ray)
    angle2 = angle_bisector_tool(C1, X, rnd_point_on_ray)

    g = env.cur_goal()[0]

    if angle1.identical_to(g):
        C = C0
    if angle2.identical_to(g):
        C = C1

    return [
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [rnd_point_on_ray, X]),
        construction.ConstructionProcess('Angle_Bisector', [C, X, rnd_point_on_ray]),
        construction.ConstructionProcess('Angle_Bisector', [C, X, rnd_point_on_ray]),
    ], [
        rnd_point_on_ray,
        circle_tool(X, rnd_point_on_ray),
        circle_tool(rnd_point_on_ray, X),
        C,
        angle_bisector_tool(C, X, rnd_point_on_ray)
    ]
