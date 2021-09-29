from py_euclidea.constructions import *
import py_euclidea.ConstructionProcess as construction

def init(env):
    c0 = env.add_free_circ((232.5, 256), 119)
    c1 = env.add_free_circ((436, 267), 86)
    env.add_free_line((347, 0), (354.5, 490.5))
    env.set_tools("intersection", "Point")
    env.goal_params(c0, c1)

def construct_goals(c0, c1):
    return (
        intersection_tool(c0, c1),
    )

def get_construction(env, obj):
    circle_1 = obj[2]
    circle_2 = obj[5]
    l = obj[8]
    intersection = intersection_cc(circle_1, circle_2)
    res = [
        Point(p) for p in intersection]
    # sometimes intersection colide and we cant pick which we want, we add those intersection into construction,
    # to make construction degenerated if they are too close
    #for i in intersection_tool(l, circle_1):
     #   res.append(i)
    #for i in intersection_tool(l, circle_2):
     #   res.append(i)
    return [
               construction.ConstructionProcess('Point', [Point(p)]) for p in intersection], res

def get_tool_hints():
    return [
        "intersection",
    ]
