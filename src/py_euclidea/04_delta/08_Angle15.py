from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    _,_,ray = env.add_free_ray((230.0, 268.0), (634.5, 268.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(ray)

def construct_goals(ray):
    vecs = (rotate_vector(ray.v, ang) for ang in (-np.pi/12, np.pi/12))
    X = ray.start_point
    return tuple(
        (Ray(X, v),)
        for v in vecs
    )

def additional_bb(ray, goal):
    return Point(ray.start_point + ray.v + goal.v)

def get_construction(env, obj):
    goal = env.goals[env.goal_index][0]
    ray = [obj[i] for i in env.goal_par_indices][0]
    A = Point(ray.start_point)
    rnd_pt = Point(A.a + ray.v * 200)
    c1 = circle_tool(rnd_pt, A)
    B = Point(A.a + (rnd_pt.a - A.a)*2)
    c2 = circle_tool(B, rnd_pt)
    pts = intersection_tool(c1, c2)
    for i in pts:
        res = angle_bisector_tool(i, A, B)
        if same_line(goal, res):
            return [
                       construction.ConstructionProcess('Circle', [rnd_pt, A]),
                       construction.ConstructionProcess('Circle', [rnd_pt, A]),
                       construction.ConstructionProcess('Circle', [B, rnd_pt]),
                       construction.ConstructionProcess('Circle', [B, rnd_pt]),
                       construction.ConstructionProcess('Angle_Bisector', [i, A, B]),
                       construction.ConstructionProcess('Angle_Bisector', [i, A, B]),
                   ], [
                       rnd_pt,
                       c1,
                       B,
                       c2,
                       res,
                       i
                   ]
    raise Exception("cannot get construction of this level")

