from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(175.5, 315.0)
    X = env.add_free(259.5, 196.6, hidden=True)
    Y = env.add_free(349.9, 317.0, hidden=True)
    r1 = env.add_ray(A, X)
    r2 = env.add_ray(A, Y)
    env.set_tools(
        "Angle_Bisector","Point"
    )
    env.goal_params(A, X, Y, r1, r2)

def additional_bb(A, X, Y,r1,r2, goal):
    x = ray_tool(A,X).v
    y = ray_tool(A,Y).v
    return Point(A.a + x + y)

def construct_goals(A, X, Y,r1,r2):
    return angle_bisector_tool(X, A, Y)

def additional_degeneration(A,X,Y, r1, r2, goal):
    cpx1 = complex(*r1.v)
    cpx2 = complex(*r2.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
    return angle < 30 or angle > 90

def scale_able():
    return False

def get_construction(env, obj):
    A, X, Y,r1,r2 = [obj[i] for i in env.goal_par_indices]
    rnd_point_on_ray_X = Point(A.a + r1.v * 150)
    rnd_point_on_ray_Y = Point(A.a + r2.v * 150)

    return [

        construction.ConstructionProcess('Angle_Bisector', [rnd_point_on_ray_X, A, rnd_point_on_ray_Y]),
        construction.ConstructionProcess('Angle_Bisector', [rnd_point_on_ray_X, A, rnd_point_on_ray_Y]),
        construction.ConstructionProcess('Angle_Bisector', [rnd_point_on_ray_X, A, rnd_point_on_ray_Y]),
    ], [
        rnd_point_on_ray_X,
        rnd_point_on_ray_Y,
        angle_bisector_tool(rnd_point_on_ray_X, A, rnd_point_on_ray_Y)
    ]
