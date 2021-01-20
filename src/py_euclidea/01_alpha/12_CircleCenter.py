from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction
import numpy as np
#TODO: there is better solution using less expensive operation
# given that to do perp bisector u need 2 circles and one line

def init(env):
    circ = env.add_free_circ((312.75, 251.75), 113.75)
    env.set_tools(
        "move", "Point", "Line", "Circle",
        "Perpendicular_Bisector", "intersection",
    )
    env.goal_params(circ)

def construct_goals(circ):
    return Point(circ.c)

def get_construction(env, obj):
    circle = obj[2]
    p1, p2, p3, _ = main_circle_sectors(circle)

    return [
        #construction.ConstructionProcess('Point', [p1]),
        #construction.ConstructionProcess('Point', [p2]),
        construction.ConstructionProcess('Perpendicular_Bisector', [p1, p2]),
        construction.ConstructionProcess('Perpendicular_Bisector', [p1, p2]),
        construction.ConstructionProcess('Perpendicular_Bisector', [p1, p2]),
        #construction.ConstructionProcess('Point', [p3]),
        construction.ConstructionProcess('Perpendicular_Bisector', [p3, p1]),
        construction.ConstructionProcess('Perpendicular_Bisector', [p3, p1]),
        construction.ConstructionProcess('Point', [Point(circle.c)]),
    ], [
        p1,
        p2,
        p3,
        perp_bisector_tool(p1, p2),
        perp_bisector_tool(p3, p1),
        Point(circle.c)
    ]
