from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    A = env.add_free(232.0, 251.0)
    B = env.add_free(351.5, 250.0)

    env.set_tools(
        "move", "Point", "Circle",
        "intersection",
    )
    env.goal_params(A, B)

def construct_goals(A, B):
    # Other goal is rly hard to describe only by the image, so we commented it for multi-level purposes
    return [
        (Point(2*B.a - A.a),),
        #(Point(-2*B.a + 3*A.a),),
    ]

def get_construction(env, obj):
    goal = env.goals[env.goal_index]
    A = obj[0]
    B = obj[1]
    c1 = circle_tool(A, B)
    c2 = circle_tool(B, A)
    P1, P2 = intersection_tool(c1, c2)

    return [
        construction.ConstructionProcess('Circle', [A, B]),
        construction.ConstructionProcess('Circle', [B, A]),
        construction.ConstructionProcess('Circle', [P1, P2]),
        construction.ConstructionProcess('Circle', [P1, P2]),
        construction.ConstructionProcess('Circle', [P1, P2]),
        construction.ConstructionProcess('Point', [goal[0]]),
    ], [
        c1,
        c2,
        P1,
        P2,
        circle_tool(P1, P2),
        goal[0]
    ]
