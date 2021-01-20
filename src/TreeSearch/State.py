from enviroment_utils import  *

class State:
    def __init__(self, multilevel, all_objects, build_image=True):
        self.all_objects_len = len(all_objects)
        self.env_objects_len = len(multilevel.cur_env.objs)
        self.image = None
        if build_image:
            self.image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])



