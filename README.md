# Learning-to-solve-geometric-construction-problems-from-images
This project is based on authors master thesis.
## abstract:
Geometric constructions using ruler and compass are being solved for thousands of years. Humans are capable of solving these problems without explicit knowledge of the analytical models of geometric primitives present in the scene. On the other hand, most methods for solving these problems on a computer require an analytical model. In this thesis, we introduce a method for solving geometrical constructions with access only to the image of the given geometric construction. The method utilizes Mask R-CNN, a convolutional neural network for detection and segmentation of objects in images and videos. Outputs of the Mask R-CNN are masks and bounding boxes with class labels of detected objects in the input image. In this work, we employ and adapt the Mask R-CNN architecture to solve geometric construction problems from image input. We create a process for obtaining geometric construction steps from masks obtained from Mask R-CNN, and we describe how to train the Mask R-CNN model to solve geometric construction problems. However, solving geometric problems this way is challenging, as we have to deal with object detection and construction ambiguity. There is possibly an infinite number of ways to solve a geometric construction problem. Furthermore, the method should be able to solve problems not seen during the training. 
To solve unseen construction problems, we develop a tree search procedure that searches the space of hypothesis provided by the Mask R-CNN model. We describe multiple components of this model and experimentally demonstrate their benefits. As experiments show, our method can learn constructions of multiple problems with high accuracy. When the geometric problem is seen at training time, the proposed approach learns to solve all 68 geometric construction problems from the first six level packs of the geometric game Euclidea with an average accuracy of 92\%. The proposed approach can also solve new geometric problems unseen at training. In this significantly harder set-up, it successfully solves 31 out of these 68 geometric problems.

![Diagram of the approach](/latex_files/img/approach_schema.png)

## Project contents
The thesis with full description of the problem can be found [here](https://github.com/mackej/Learning-to-solve-geometric-construction-problems-from-images/blob/main/MackeJ_earning_to_solve_geometric_construction_problems_from_images.pdf)
This project has 4 main modules:
* Mask R-CNN scripts for training and testing
* Hypothesis explorer
* Hypothesis tree search
* Exhaustive search


Training and result evaluation for this project were run at the Slurm cluster. Some folder also contains the Slurm job assignments. Every module can be run manually.
However, inspecting the Slurm files (.sh) gives a good example of how to run the modules.

## Mask R-CNN:
This module is in folder "/src/MaskRCNNExperiment", the folder contains a module for training "/src/MaskRCNNExperiment/TrainGeometryDataset.py",
and a module for testing "/src/MaskRCNNExperiment/TestModel.py". Note that the module for the training also run a test after the training. 
This modul has a lot of input arguments,  "python TestModel.py -h" and "python TrainGeometryDataset.py -h" to see argument documentation.
## Hypothesis explorer:
This module is a program for interactive hypothesis explorer. The program can be run with: "/src/HypothesisExplorer/main.py".
However, this model requires models for hypothesis proposals. Part of the thesis is attachments with models for Alpha to Zeta levels 
(As of now models cannot be downloaded) (see the thesis Chapter 5). For the hypothesis explorer, to run properly copy models from the attachment of the thesis into "src/logs",
for further details see attachment readme file. Any model trained with our architecture can be used as a hypothesis proposal generator.
To add a new model, add its path and name to "/src/HypothesisExplorer/models_config.py". 
## Hypothesis tree search:
As hypothesis explorer, hypothesis tree search also requires models for hypothesis proposals. To run see scripts:
"run_whole_level_pack_inference.py" , "run_model_test_with_tree_search.py", "run_hypothesis_tree_search.py" in folder "/src/HypothesisExplorer"
## Exhaustive search:
This module is in folder "/src/TreeSearch". This module is an exhaustive search of geometric problems. To run this search, see the file 
"/src/TreeSearch/solve_TreeSearch.py". This module also contains several scripts for the evalution of branching factor estimates (see the thesis Chapter 2).
### Other	
The Euclidea-like environment can be found in "/src/py_euclidea". In folder "/src/mrcnn" is modified Matterport implementation of the Mask R-CNN.
