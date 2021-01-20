from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np

def init(env):
    A,ray1,ray2 = env.add_free_angle(
        (230.0, 268.0), (621.5, 268.0), (602.5, 115.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "intersection",
    )
    env.goal_params(ray1, ray2)

def construct_goals(ray1, ray2):
    result = []
    for (rx, ry) in (ray1, ray2), (ray2, ray1):
        n = ry.n * np.dot(ry.n, rx.v)
        v = rx.v - 2*n
        result.append((Ray(rx.start_point, v),))
    return result

#def additional_bb(ray1, ray2, goal):
    #if np.linalg.matrix_rank((ray2.v+goal.v, ray1.v)) <= 1:
    #    ray = ray1
    #else: ray = ray2
    #return Point(ray.start_point + ray.v)

def additional_degeneration(ray1, ray2, goal):
    cpx1 = complex(*ray1.v)
    cpx2 = complex(*ray2.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
    return angle < 20 or angle > 60

def get_construction(env, obj):
    g = env.cur_goal()[0]
    r1 = obj[2]
    r2 = obj[4]
    cpx1 = complex(*r1.v)
    cpx2 = complex(*r2.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))

    X = Point(r1.start_point)

    rnd_point_on_ray = Point(X.a + r1.v * 200)

    c = circle_tool(X, rnd_point_on_ray)

    P1, _ = intersection_tool(c, r1)
    P2, _ = intersection_tool(c, r2)

    angle_c1 = circle_tool(P1, P2)
    angle_c2 = circle_tool(P2, P1)

    D1, D2 = intersection_tool(c, angle_c1)
    D3, D4 = intersection_tool(c, angle_c2)



    if is_point_on(D1.a, g):
        final_D = D1
        final_P1 = P1
        final_P2 = P2
    if is_point_on(D2.a, g):
        final_D = D2
        final_P1 = P1
        final_P2 = P2
    if is_point_on(D3.a, g):
        final_D = D3
        final_P1 = P2
        final_P2 = P1
    if is_point_on(D4.a, g):
        final_D = D4
        final_P1 = P2
        final_P2 = P1

    return [
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [X, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [final_P1, final_P2]),
        construction.ConstructionProcess('Circle', [final_P1, final_P2]),
        construction.ConstructionProcess('Line', [final_D, X]),
        construction.ConstructionProcess('Line', [final_D, X]),
    ], [
        rnd_point_on_ray,
        circle_tool(X, rnd_point_on_ray),
        circle_tool(final_P1, final_P2),
        final_D,
        line_tool(final_D, X)
    ]
