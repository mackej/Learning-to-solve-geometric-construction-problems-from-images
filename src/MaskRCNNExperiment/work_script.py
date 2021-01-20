import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import TrainingUtils

#TrainingUtils.clump_csv_resuts("../logs/Epsilon_One_by_One", pattern='20200929')

#for i in os.listdir("E:/MFF-SchoolStuff/Master-Thesis/src/logs"):
#TrainingUtils.events_to_pyplot(os.path.join("E:/MFF-SchoolStuff/Master-Thesis/src/logs",i))
#TrainingUtils.plot_multiple_losses("../logs/Delta_as_whole",regex=".*",  loss="loss")
TrainingUtils.plot_multiple_losses("../logs/redoing_alpha", regex="(alpha_with_4+|alpha_whithout_4+)$")
#TrainingUtils.plot_multiple_losses("../logs/redoing_alpha", regex="(alpha_again|alpha_whole_nn_training_2|alpha_4\+_included)$" , loss="mrcnn_class_loss")
#TrainingUtils.plot_multiple_losses("../logs/redoing_alpha", regex="(alpha_again|alpha_whole_nn_training_2|alpha_4\+_included)$" , loss="mrcnn_bbox_loss")
#TrainingUtils.plot_multiple_losses("../logs/redoing_alpha", regex="(alpha_again|alpha_whole_nn_training_2|alpha_4\+_included)$" , loss="mrcnn_mask_loss")