from py_euclidea.constructions import *
import itertools


class EnvStep:
    def __init__(self, args_i, otypes):
        self.args_i = args_i
        self.otypes = singleton_to_tuple(otypes)
        for i in args_i: assert (isinstance(i, int))
        for t in self.otypes: assert (isinstance(t, type))
        self.none_res = tuple(None for _ in self.otypes)

    def run(self, env):
        args = tuple(env.objs[i] for i in self.args_i)
        if None in args: return self.none_res
        try:
            result = self.constr(*args)
        except:
            return self.none_res

        if not isinstance(result, (list, tuple)):
            result = result,
        # Macke J: if we are using intersection then in most cases result is padded with None's to match with
        # expected number of outputs, but sometimes its not. possible error in run of the step.
        # TODO: fix self.constr(*args) and delete next 2 lines
        if len(result) == 1 and len(self.otypes) == 2:
            l = list(result)
            l.append(None)
            result = tuple(l)

        assert (len(result) == len(self.otypes))
        for x, t in zip(result, self.otypes):
            assert (isinstance(x, t) or x is None)
        return result


class ConstrStep(EnvStep):
    def __init__(self, constr, args_i, otypes):
        EnvStep.__init__(self, args_i, otypes)
        self.constr_f = constr

    def constr(self, *args):
        return self.constr_f(*args)


class MovableStep(EnvStep):
    def __init__(self, coor_to_obj, coor, args_i, otypes):
        EnvStep.__init__(self, args_i, otypes)
        self.coor = coor
        self.coor_to_obj = coor_to_obj

    def constr(self, *args):
        return self.coor_to_obj(self.coor, *args)


class Environment:
    point_min_dist = 30
    def __init__(self, win_size):
        self.corners = np.array(((0, 0), win_size), dtype=float)
        self.steps = []
        self.rand_steps = []
        self.visible = set()
        self.obj_to_movable = dict()
        self.obj_num = 0
        self.goal_index = 0
        self.construction = None
        self.construction_objects = None
        self.construction_steps = None

    def next_goal(self):
        self.goal_index += 1
        if self.goal_index >= len(self.goals):
            self.goal_index = 0

    def cur_goal(self):
        return self.goals[self.goal_index]

    def add(self, step, hidden=False):
        step.hidden = hidden
        self.steps.append(step)
        next_obj_num = self.obj_num + len(step.otypes)
        output = list(range(self.obj_num, next_obj_num))
        step.output = output
        self.obj_num = next_obj_num
        if not hidden: self.visible.update(output)
        return output

    def run_step(self, step):
        output = step.run(self)
        for obj in output:
            if obj is not None:
                obj.index = len(self.objs)
                obj.hidden = step.hidden
            self.objs.append(obj)

    def finish_goal(self):
        goal_par_objs = [self.objs[i] for i in self.goal_par_indices]
        goal = singleton_to_tuple(self.generate_goal(*goal_par_objs))
        if len(goal) > 0 and not isinstance(goal[0], (list, tuple)):
            goal = (goal,)
        self.goals = goal

    def run_steps(self):
        self.objs = []
        for step in self.steps:
            self.run_step(step)
        self.finish_goal()

    def check_goal(self):
        for goal in self.goals:
            for subgoal in goal:
                if not any(
                        obj.identical_to(subgoal)
                        for obj in self.visible_objs()
                ): break
            else:
                return True
        return False

    def add_and_run(self, step):
        output = step.run(self)
        if all(out is None for out in output):
            return False
        self.add(step)
        for obj in output:
            if obj is not None: obj.index = len(self.objs)
            self.objs.append(obj)
        return True

    # adding specific objects

    def add_movable(self, coor_to_obj, coor, args, otypes, hidden=False):
        step = MovableStep(coor_to_obj, coor, args, otypes)
        output = self.add(step, hidden)
        for i in output: self.obj_to_movable[i] = step
        return tuple_to_singleton(output)

    def add_constr(self, constr, args, otypes, hidden=False):
        output = self.add(ConstrStep(constr, args, otypes), hidden)
        return tuple_to_singleton(output)

    def add_rand_init(self, objs, initializer, args=(), kwargs={}):
        objs = singleton_to_tuple(objs)
        self.rand_steps.append((
            tuple(self.obj_to_movable[obj] for obj in objs),
            initializer, args, kwargs,
        ))

    def add_free(self, x_or_coor, y=None, hidden=False, rand_init=True):
        if y is None:
            coor = x_or_coor
        else:
            coor = x_or_coor, y
        res = self.add_movable(Point, np.array(coor), (), Point, hidden)
        if rand_init: self.add_rand_init(res, random_point)
        return res

    def add_dep(self, coor, obj, hidden=False, rand_init=True):
        res = self.add_movable(point_on, np.array(coor), (obj,), Point, hidden)
        if rand_init: self.add_rand_init(res, random_point_on, args=(obj,))
        return res

    def add_line(self, p1, p2, hidden=False):
        return self.add_constr(line_tool, (p1, p2), Line, hidden)

    def add_perp(self, p, l, hidden=False):
        if isinstance(l, (tuple, list)):
            l = self.add_line(*l, hidden=True)
        return self.add_constr(perp_tool, (l, p), Line, hidden)

    def add_circle(self, C, X, hidden=False):
        return self.add_constr(circle_tool, (C, X), Circle, hidden)

    def add_segment(self, A, B, hidden=False):
        return self.add_constr(segment_tool, (A, B), Segment, hidden)

    def add_ray(self, A, B, hidden=False):
        return self.add_constr(ray_tool, (A, B), Ray, hidden)

    def add_free_line(self, A_coor, B_coor, hidden=False):
        A = self.add_free(A_coor, hidden=True)
        B = self.add_free(B_coor, hidden=True)
        line = self.add_line(A, B)
        return line

    def add_free_circ(self, C_coor, radius, hidden=False,
                      hidden_center=True, rand_init=None):
        C_coor = np.array(C_coor)
        v = np.array((-0.4, 1.1))
        v /= np.linalg.norm(v)
        X_coor = C_coor + v * radius
        C = self.add_free(C_coor, hidden=hidden_center, rand_init=(rand_init is None))
        X = self.add_free(X_coor, hidden=True, rand_init=(rand_init is None))
        circ = self.add_circle(C, X, hidden)
        if rand_init is not None:
            if isinstance(rand_init, (tuple, list)):
                circ_gen, *args = rand_init
            else:
                circ_gen, args = rand_init, ()

            def f(*args):
                center, radius = circ_gen(*args)
                return center, center + v * radius

            self.add_rand_init((C, X), f, args)
        return circ

    def add_free_segment(self, A_coor, B_coor, hidden=False):
        A = self.add_free(A_coor, hidden=hidden)
        B = self.add_free(B_coor, hidden=hidden)
        segment = self.add_segment(A, B, hidden=hidden)
        return A, B, segment

    def add_free_ray(self, A_coor, B_coor, hidden=False):
        A = self.add_free(A_coor, hidden=hidden)
        B = self.add_free(B_coor, hidden=True)
        ray = self.add_ray(A, B, hidden=hidden)
        return A, B, ray

    def add_free_angle(self, A_coor, X_coor, Y_coor, hidden=False):
        A, X, ray_x = self.add_free_ray(A_coor, X_coor, hidden=hidden)
        Y = self.add_free(Y_coor, hidden=True)
        ray_y = self.add_ray(A, Y, hidden=hidden)
        return A, ray_x, ray_y

    def add_parallelogram(self, A, B, C, hidden=False):
        return self.add_constr(
            parallelogram, (A, B, C),
            (Point, Line, Line, Line, Line),
            hidden=hidden,
        )

    def add_free_triangle(self, A_coor, B_coor, C_coor, hidden=False, acute_prob=0.6):
        A = self.add_free(A_coor, hidden=hidden, rand_init=False)
        B = self.add_free(B_coor, hidden=hidden, rand_init=False)
        C = self.add_free(C_coor, hidden=hidden, rand_init=False)
        a = self.add_segment(B, C, hidden=hidden)
        b = self.add_segment(C, A, hidden=hidden)
        c = self.add_segment(A, B, hidden=hidden)
        self.add_rand_init((A, B, C), random_triangle, kwargs={"acute_prob": acute_prob})
        return (A, B, C), (a, b, c)

    def add_free_rectangle(self, A_coor, B_coor, C_coor, hidden=False):
        A = self.add_free(A_coor, hidden=hidden)
        B = self.add_free(B_coor, hidden=hidden)
        perp = self.add_perp(B, (A, B), hidden=True)
        C = self.add_dep(C_coor, perp, hidden=hidden)
        D, *segments = self.add_parallelogram(A, B, C, hidden=hidden)
        return (A, B, C, D), segments

    def add_free_trapezoid(self, A_coor, B_coor, C_coor, D_coor, hidden=False):
        A = self.add_free(A_coor, hidden=hidden, rand_init=False)
        B = self.add_free(B_coor, hidden=hidden, rand_init=False)
        C = self.add_free(C_coor, hidden=hidden, rand_init=False)
        a = self.add_segment(A, B)
        para = self.add_constr(
            parallel_tool, (a, C), Line, hidden=True)
        D = self.add_dep(D_coor, para, hidden=hidden, rand_init=False)
        b = self.add_segment(B, C)
        c = self.add_segment(C, D)
        d = self.add_segment(D, A)
        self.add_rand_init((A, B, C, D), random_trapezoid)
        return (A, B, C, D), (a, b, c, d)

    def add_free_square(self, A_coor, B_coor, hidden=False):
        A = self.add_free(A_coor, hidden=hidden)
        B = self.add_free(B_coor, hidden=hidden)
        C, D, *segments = self.add_constr(
            square, (A, B),
            (Point, Point, Segment, Segment, Segment, Segment),
            hidden=hidden
        )
        return (A, B, C, D), segments

    # final tweaks

    def set_tools(self, *tools, start=None):
        self.enabled_tools = set(tools)
        if start is not None:
            assert (start in self.enabled_tools)
            self.start_tool = start
        elif "circle" in self.enabled_tools:
            self.start_tool = "circle"
        elif "line" in self.enabled_tools:
            self.start_tool = "line"
        else:
            self.start_tool = tools[0]

    def goal_params(self, *params):
        self.goal_par_indices = params
        self.min_steps = len(self.steps)
        self.min_objs = self.obj_num

    # Selecting objects

    def visible_objs(self):
        return (self.objs[n] for n in self.visible if self.objs[n] is not None)

    def objs_of_type(self, obj_type, include_goal=False):
        if include_goal:
            objs = filter(lambda obj: isinstance(obj, obj_type), itertools.chain(self.visible_objs(), self.cur_goal()))
        else:
            objs = filter(lambda obj: isinstance(obj, obj_type), self.visible_objs())
        return objs

    def points(self):
        return self.objs_of_type(Point)

    def point_sets(self):
        return self.objs_of_type(PointSet)

    def closest_obj_of_type(self, coor, obj_type, include_goal=False):

        obj_dist = [
            (obj.dist_from(coor), obj)
            for obj in self.objs_of_type(obj_type, include_goal=include_goal)
        ]
        if not obj_dist: return 0, None
        dist, obj = min(obj_dist, key=lambda x: x[0])
        return dist, obj

    def closest_point(self, coor):
        return self.closest_obj_of_type(coor, Point)

    def closest_set(self, coor):
        return self.closest_obj_of_type(coor, PointSet)

    # other operations
    def pop(self):
        if len(self.steps) <= self.min_steps:
            print("no more steps to go back")
            return
        step = self.steps.pop()
        prev_obj_num = self.obj_num - len(step.otypes)
        output = range(prev_obj_num, self.obj_num)
        self.visible.difference_update(output)
        self.obj_num = prev_obj_num
        del self.objs[prev_obj_num:]

    def restart(self):
        self.visible.difference_update(range(self.min_objs, self.obj_num))
        self.obj_num = self.min_objs
        del self.objs[self.obj_num:]
        del self.steps[self.min_steps:]

    # random initialization

    def get_bounding_box(self):
        bb_args = [self.objs[i] for i in self.goal_par_indices]
        bb_objs = self.additional_bb(*(bb_args + [tuple_to_singleton(self.cur_goal())]))
        bb_objs = singleton_to_tuple(bb_objs)
        construction_bbs = []
        for i in self.construction:
            for construction_point in i.action_clicks:
                construction_bbs.append(construction_point)
        objs = itertools.chain(self.visible_objs(), self.cur_goal(), bb_objs, construction_bbs)

        obj_corners = filter(
            lambda x: x is not None,
            (obj.get_bounding_box() for obj in objs)
        )
        lu, rd = list(zip(*obj_corners))
        if not lu: return None

        lu = np.min(np.stack(lu), axis=0)
        rd = np.max(np.stack(rd), axis=0)
        return np.stack((lu, rd))

    def rnd_init(self, offset=50, min_ratio=0.3):
        self.construction_steps = None
        exception = None
        for i in range(10000):
            try:
                step_iter = iter(self.steps)
                self.objs = []
                for target, initializer, args, kwargs in self.rand_steps:
                    if args:
                        max_arg = max(args)
                        while len(self.objs) <= max_arg:
                            self.run_step(next(step_iter))
                        args = tuple(self.objs[i] for i in args)
                    coors = initializer(*args, **kwargs)
                    if len(target) == 1: coors = coors,
                    for coor, targ in zip(coors, target):
                        targ.coor = coor

                for step in step_iter: self.run_step(step)
                self.finish_goal()
                if len(self.goals) == 0: continue

                self.goal_index = np.random.randint(len(self.goals))

                self.construction, self.construction_objects = self.generate_construction(self, self.objs)
            except Exception as e:
                exception = e
                continue

            bb = self.get_bounding_box()
            corners = self.corners + np.array(
                ((offset, offset), (-offset, -offset)))
            if self.scale_able:
                max_scale = min((corners[1] - corners[0]) /
                            np.abs(bb[1] - bb[0]))
                min_scale = max(1, self.get_min_scale())
                if min_scale > max_scale:
                    continue
                scale = np.random.uniform(min_scale, max_scale)
            else:
                scale = 1

            if self.is_degenerated(
                    scale=scale, center=(bb[0] + bb[1]) / 2
            ):
                continue

            check_args = [self.objs[i] for i in self.goal_par_indices]
            if not self.ini_check(*(
                    check_args + [tuple_to_singleton(self.cur_goal()), scale]
            )):
                continue

            break
        else:
            print("Failed to find a construction")
            if exception is not None: raise exception

        min_pos, max_pos = corners - (bb * scale)

        pos = np.random.uniform(min_pos, max_pos)


        for step in self.steps:
            if isinstance(step, MovableStep):
                step.coor = step.coor * scale + pos

        obj_set = set(itertools.chain(self.construction_objects, self.objs, *self.goals))
        for obj in obj_set:
            obj.scale(scale)
            obj.shift(pos)
        for a in self.construction:
            for c in a.action_clicks:
                c.scale(scale)
                c.shift(pos)
        if self.bbox_dependant:
            self.construction, self.construction_objects = self.generate_construction(self, self.objs, corners=corners)



    def get_min_scale(self):
        '''
        min scale is based on minimal point distance. This method does-not return actual minimal scale, but rather
        its lower bound which is minimal_poin_distance / MIN(|AB|) for A,B in points. This criteria is most often used
        degeneration rule,
        when we do not use this min_scale
        :return:
        '''
        points = []
        for obj in itertools.chain(self.visible_objs(), self.construction_objects):
            if isinstance(obj, Point):
                points.append(obj)
        if len(points) < 2:
            return 1
        min = float('inf')
        for A, B in itertools.combinations(points, 2):
            d = np.sum((A.a - B.a) ** 2)
            if d < min:
                min = d
        return np.sqrt(self.point_min_dist) / min

    def is_degenerated(self, scale, center):
        deg_args = [self.objs[i] for i in self.goal_par_indices]
        degenerated = self.additional_degeneration(*(deg_args + [tuple_to_singleton(self.cur_goal())]))
        if degenerated:
            return True
        points = []
        lines = []
        circles = []

        # Construction also returns new object that will be created during construction,
        # In prev version we were using cur_goal as part of degeneration, now we just use this construction

        for obj in itertools.chain(self.visible_objs(), self.construction_objects):
            if isinstance(obj, Point):
                points.append(obj)
            elif isinstance(obj, Circle):
                circles.append(obj)
            elif isinstance(obj, Line):
                lines.append(obj)
            else:
                raise Exception("test_degenecary: Unexpected type {}".format(type(obj)))

        # too small circles / segments

        segment_min_len = 50 / scale
        circle_min_radius = 30 / scale
        for l in lines:
            if isinstance(l, Segment):
                A, B = l.end_points
                if np.linalg.norm(A - B) < segment_min_len:
                    return True
        for c in circles:
            if c.r < circle_min_radius:
                return True

        # object too close to another

        point_min_dist = self.point_min_dist / scale
        point_min_dist_sq = point_min_dist ** 2
        circle_min_dist = 30 / scale
        line_min_sang = 0.05
        line_min_dist = 30 / scale
        seg_line_min_dist = 30 / scale

        # Add construction points to points to determine if construction is degenerated
        # for construction_step in self.construction:
        #   for A,B in itertools.combinations(construction_step.action_clicks,2):
        #      if A.dist_from(B.a) < point_min_dist: return True
        '''
        # New method of checking degeneration in tasks, we generate whole construction so we can
        # check newly created objects lines, circles not just points like in previous method(below)
        construction_points = [p.a for p in points]
        for construction_step in self.construction:
            for p in construction_step.action_clicks:
                # numpy does not have np.anyclose equivalent to np.allclose .... next line is np.anyclose
                if not np.any(np.sum(np.isclose(construction_points, p.a), axis=1) == 2):
                    construction_points.append(p.a)
                    points.append(Point(p.a))
        '''

        # main objects
        for A, B in itertools.combinations(points, 2):
            if np.sum((A.a - B.a) ** 2) < point_min_dist_sq:
                return True
        for c1, c2 in itertools.combinations(circles, 2):
            if np.linalg.norm(c1.c - c2.c) + abs(c1.r - c2.r) < circle_min_dist:
                return True

        for l1, l2 in itertools.combinations(lines, 2):
            if isinstance(l1, Segment) or isinstance(l2, Segment):
                if not isinstance(l1, Segment): l1, l2 = l2, l1
                if not isinstance(l2, Segment) and not isinstance(l2, Ray) and all(l1.v == l2.v):
                    # this occurs in construction when we need input segment to be line
                    continue
                if all(l2.dist_from(x) < seg_line_min_dist for x in l1.end_points):
                    return True
            if abs(np.cross(l1.n, l2.n)) < line_min_sang:
                ddist1 = np.dot(l1.n, center) - l1.c
                ddist2 = np.dot(l2.n, center) - l2.c
                if np.dot(l1.n, l2.n) < 0: ddist2 *= -1
                if abs(ddist1 - ddist2) < line_min_dist:
                    return True

        # get all other intersections and check weather they are close to rest of the points, points in
        # variable point are also checked with each other, those point can be close to each other,
        # but cannot be close to points in construction / inputs
        important_points = np.array([p.a for p in points])

        for i, j in itertools.combinations(itertools.chain(lines, circles), 2):
            intersections = intersection_tool(i, j, exception_on_fail=False)
            if not isinstance(intersections, list):
                intersections = [intersections]
            for pt in intersections:
                if pt is None:
                    continue
                if not np.any(np.sum(np.isclose(important_points, pt.a), axis=1) == 2):
                    # check weather pt inst too close to some construction point
                    dist = np.sum((important_points - pt.a) ** 2, axis=1)
                    if np.min(dist) < point_min_dist_sq:
                        return True
        return False

    def get_construction_steps(self):
        if self.construction_steps is not None:
            return self.construction_steps
        reduced_construction = [self.construction[0]]
        for i in self.construction:
            if not (
                    reduced_construction[-1].tool_name == i.tool_name and
                    len(i.action_clicks) == len(reduced_construction[-1].action_clicks) and
                    np.all([
                        np.all(reduced_construction[-1].action_clicks[p].a == i.action_clicks[p].a)
                        for p in range(len(i.action_clicks))])
            ):
                reduced_construction.append(i)

        # for c in reduced_construction:

    def get_copy_of_state_objects(self):
        objs_copy = []
        for ob in self.objs:
            if not hasattr(ob, 'hidden') or ob.hidden is False:
                objs_copy.append(ob.copy())
        return objs_copy


def load_level(level_pack, level, win_size):
    import importlib

    module = importlib.import_module(level_pack + '.' + level)
    env = Environment(win_size)
    env.generate_goal = module.construct_goals
    if hasattr(module, "additional_bb"):
        env.additional_bb = module.additional_bb
    else:
        env.additional_bb = lambda *args: ()

    if hasattr(module, "additional_degeneration"):
        env.additional_degeneration = module.additional_degeneration
    else:
        env.additional_degeneration = lambda *args: ()

    if hasattr(module, "scale_able"):
        env.scale_able = module.scale_able()
    else:
        env.scale_able = True

    if hasattr(module, "bbox_dependant"):
        env.bbox_dependant = module.bbox_dependant()
    else:
        env.bbox_dependant = False

    if hasattr(module, "ini_check"):
        env.ini_check = module.ini_check
    else:
        env.ini_check = lambda *args: True
    module.init(env)
    env.run_steps()
    env.generate_construction = module.get_construction

    return env