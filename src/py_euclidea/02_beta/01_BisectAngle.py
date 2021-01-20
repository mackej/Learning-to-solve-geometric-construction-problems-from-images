
from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(175.5, 315.0)
    X = env.add_free(385.5, 19.0, hidden = True)
    Y = env.add_free(611.5, 320.0, hidden = True)
    r1 = env.add_ray(A, X)
    r2 = env.add_ray(A, Y)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(A, X, Y, r1,r2)

def scale_able():
    return False

def additional_degeneration(A,X,Y, r1, r2, goal):
    cpx1 = complex(*r1.v)
    cpx2 = complex(*r2.v)
    angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
    return angle < 30 or angle > 90

def construct_goals(A, X, Y, r1, r2):
    return angle_bisector_tool(X, A, Y)

def additional_bb(A, X, Y,r1,r2, goal):
    x = ray_tool(A,X).v
    y = ray_tool(A,Y).v
    return Point(A.a + x + y)

def get_construction(env, obj):
    A = obj[0]
    X = obj[1]
    Y = obj[2]
    ax = obj[3]
    ay = obj[4]

    rnd_point_on_ray = Point(A.a + ax.v * 200)
    c = circle_tool(A,rnd_point_on_ray)
    P1, _ = intersection_tool(ax, c)
    P2, _ = intersection_tool(ay, c)

    return [
        construction.ConstructionProcess('Circle', [A, rnd_point_on_ray]),
        construction.ConstructionProcess('Circle', [A, rnd_point_on_ray]),

        construction.ConstructionProcess('Perpendicular_Bisector', [P1, P2]),
        construction.ConstructionProcess('Perpendicular_Bisector', [P1, P2]),
    ], [
        rnd_point_on_ray,
        circle_tool(A, rnd_point_on_ray),
        perp_bisector_tool(P1, P2)
    ]
