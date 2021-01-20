import os
import re

class LevelSelector:
    @staticmethod
    def get_levels(match=".*", not_include=[], skip_tutorials=False):
        all = ["01_alpha", "02_beta", "03_gamma", "04_delta", "05_epsilon", "06_zeta"]
        tutorials = [
            "01_alpha.08_TPerpBisector.py",
            "02_beta.02_TBisectAngle.py",
            "02_beta.09_TDropPerp.py",
            "05_epsilon.02_TParallel.py",
            "06_zeta.06_TCircleByRadius.py"
        ]
        regex = re.compile(match, re.IGNORECASE)
        is_level = re.compile("[0-9][0-9]_[a-zA-Z0-9]*", re.IGNORECASE)

        result = []
        path = os.path.dirname(__file__) + os.path.sep +"py_euclidea"
        for level_pack in os.listdir(path):
            level_pack_path = path + os.path.sep + level_pack
            if not os.path.isdir(level_pack_path) or level_pack not in all:
                continue
            for level in os.listdir(level_pack_path):
                level_name = level[:-3]
                if level in not_include or not is_level.search(level_name):
                    continue
                full_lvl_name = level_pack + "." + level_name
                if skip_tutorials and full_lvl_name in tutorials:
                    continue
                if regex.search(full_lvl_name):
                    result.append(("py_euclidea."+level_pack, level_name))
        result.sort(key=lambda s: s[0]+s[1])
        return result

