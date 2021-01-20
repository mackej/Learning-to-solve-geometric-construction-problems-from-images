from HypothesisExplorer.explorer import *
import pickle


if __name__ == '__main__':
   h = explorer()

   models, names = h.choose_models()
   h.load_weigths(models, names)
   while True:
      h.enable_disable_models()
      levels = h.choose_levels()
      h.run_inference(levels)


