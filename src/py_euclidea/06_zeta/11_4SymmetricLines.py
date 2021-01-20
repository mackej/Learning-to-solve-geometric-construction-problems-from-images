from py_euclidea.constructions import *
import itertools
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(325.5, 258.0)
    X = env.add_free(449.0, 11.0, hidden = True)
    Y = env.add_free(623.0, 126.0, hidden = True)
    Z = env.add_free(578.5, 427.0, hidden = True)
    x = env.add_line(A,X)
    y = env.add_line(A,Y)
    z = env.add_line(A,Z)

    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "Angle_Bisector",
        "Perpendicular", "intersection", "Parallel",
        "Compass"
    )
    env.goal_params(A,x,y,z)

def construct_goals(A,in_x,in_y,in_z):
    result = []
    for x,y,z in (in_x,in_y,in_z), (in_y,in_z,in_x), (in_z,in_x,in_y):
        if np.dot(x.n, y.n) >= 0: axis = x.n+y.n
        else: axis = x.n-y.n
        axis /= np.linalg.norm(axis)
        n = z.n - 2*(axis * np.dot(z.n, axis))
        result.append((Line(n, np.dot(n, A.a)),))
    return result

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A, in_x, in_y, in_z = [obj[i] for i in env.goal_par_indices]
    (in_x, in_y, in_z) = [(in_x, in_y, in_z), (in_y, in_z, in_x), (in_z, in_x, in_y)][env.goal_index]
    rnd_pt = Point(A.a + in_x.v * 250)
    c1 = circle_tool(A, rnd_pt)
    # choose closer point for easier construction
    Y_possible_pts = intersection_tool(c1, in_y)
    Y_p = Y_possible_pts[np.argmin(np.sum(([i.a for i in Y_possible_pts] - rnd_pt.a) ** 2, axis=1))]
    Z_possible_pts = intersection_tool(c1, in_z)
    Z_p = Z_possible_pts[np.argmin(np.sum(([i.a for i in Z_possible_pts] - rnd_pt.a) ** 2, axis=1))]
    c2 = compass_tool(rnd_pt, Z_p, Y_p)
    G, G2 = intersection_tool(c1, c2)
    g = line_tool(G, A)
    if not is_this_goal_satisfied(g, goal):
        G = G2
        g = line_tool(G, A)

    return [
               construction.ConstructionProcess('Circle', [A, rnd_pt]),
               construction.ConstructionProcess('Circle', [A, rnd_pt]),
               construction.ConstructionProcess('Compass', [rnd_pt, Z_p, Y_p]),
               construction.ConstructionProcess('Compass', [rnd_pt, Z_p, Y_p]),
               construction.ConstructionProcess('Compass', [rnd_pt, Z_p, Y_p]),
               construction.ConstructionProcess('Line', [G, A]),
               construction.ConstructionProcess('Line', [G, A]),

           ], [
               rnd_pt, c1,
               Y_p, Z_p,
               c2,
               G, g, G2
           ]


def additional_degeneration(A, in_x, in_y, in_z, goal):
    # prevent lines to have lesser then 20 degree angle
    for x, y in itertools.combinations([in_x, in_y, in_z], 2):
        cpx1 = complex(*x.v)
        cpx2 = complex(*y.v)
        angle = np.abs(np.angle(cpx2 / cpx1, deg=True))
        angle = min(angle, 180-angle)
        if angle < 20:
            return True

    return False
