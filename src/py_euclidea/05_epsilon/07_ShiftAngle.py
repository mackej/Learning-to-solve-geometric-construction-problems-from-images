from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A, rx, ry = env.add_free_angle(
        (156.0, 316.5), (631.5, 328.5), (437.0, 14.0))
    X = env.add_free(310.0, 226.0, rand_init = False)
    env.add_rand_init(X, random_point_in_angle, (ry, rx))

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel"
    )
    env.goal_params(X, rx, ry)

def construct_goals(X, rx, ry):
    return (Ray(X.a, rx.v), Ray(X.a, ry.v))



def get_construction(env, obj):
    X, rx, ry = [obj[i] for i in env.goal_par_indices]
    length = np.linalg.norm(X.a - rx.start_point)
    pt_on_rx = Point(rx.start_point + (rx.v * length))
    pt_on_ry = Point(ry.start_point + (ry.v * length))
    return [
               construction.ConstructionProcess('Parallel', [pt_on_rx, X]),
               construction.ConstructionProcess('Parallel', [pt_on_ry, X]),
           ], [
               pt_on_rx, pt_on_ry,
               parallel_tool(rx, X),
               parallel_tool(ry, X),
           ]
