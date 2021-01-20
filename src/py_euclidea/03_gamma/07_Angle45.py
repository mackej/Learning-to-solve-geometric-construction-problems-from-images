from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    _,_,ray = env.add_free_ray(
        (230.0, 268.0), (623.0, 268.0))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection",
    )
    env.goal_params(ray)

def scale_able():
    return False

def construct_goals(ray):
    vecs = (rotate_vector(ray.v, ang) for ang in (-np.pi/4, np.pi/4))
    X = ray.start_point
    return tuple(
        (Ray(X, v),)
        for v in vecs
    )

def additional_bb(ray, goal):
    return Point(ray.start_point + ray.v + goal.v)

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    ray = obj[2]
    start = Point(ray.start_point)
    length = 200
    rnd_point_on_line = Point(start.a + ray.v * length)
    perp = perp_bisector_tool(rnd_point_on_line, start)
    point_on_perp = Point(start.a + (perp.v * length))
    angle_bisec = angle_bisector_tool(point_on_perp, start, rnd_point_on_line)
    if not same_line(goal[0], angle_bisec):
        point_on_perp = Point(start.a - (perp.v * length))

    return [

        construction.ConstructionProcess('Perpendicular', [rnd_point_on_line, start]),
        construction.ConstructionProcess('Angle_Bisector', [point_on_perp, start, rnd_point_on_line]),
        construction.ConstructionProcess('Angle_Bisector', [point_on_perp, start, rnd_point_on_line]),
        construction.ConstructionProcess('Angle_Bisector', [point_on_perp, start, rnd_point_on_line]),
    ], [
        perp,
        point_on_perp,
        rnd_point_on_line,
        angle_bisec
    ]
