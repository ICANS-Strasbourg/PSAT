# PSAT: **P**ediatric **S**egmentation **A**pproaches via **A**dult **A**ugmentations and **T**ransfer

### PSAT Notation and Meaning

**PSAT** stands for **Plan, Set, Augmentation, and Tuning**. Each letter represents a key component of the training configuration for segmentation approaches in pediatric radiotherapy. Below is a breakdown of the notation and its possible options:

| **Component**   | **Symbol**   | **Options and Description**                                                                                   |
|------------------|--------------|----------------------------------------------------------------------------------------------------------------|
| **Plan**         | $P$      | - **p**: Pediatric Plans  <br> - **a**: Adult Plans  <br> - **m**: Mixed Plans                                |
| **Set**          | $S$      | - **p**: Pediatric Dataset  <br> - **a**: Adult Dataset  <br> - **m**: Mixed Dataset                         |
| **Augmentation** | $A$      | - **d**: Default Augmentations  <br> - **c**: Contraction Augmentations                                      |
| **Tuning**     | $T$      | - **p**: Pediatric Fine-Tuning  <br> - **m**: Mixed Fine-Tuning  <br> - **None**: No Fine-Tuning |

---

### Example of PSAT Notation

- $P_a S_a A_d T_p$
  - **Plan**: Adult Plans  
  - **Set**: Adult Dataset  
  - **Augmentation**: Default Augmentations  
  - **Tuning**: Pediatric Fine-Tuning  

## Documentation

- [nnUNet](nnUNet/nnUNet.md)
- [Dataloading](nnUNet/dataloading/dataloading.md)
- [Preprocessing](nnUNet/preprocessing/preprocessing.md)
- [Training](nnUNet/training/training.md)
- [Notebooks](notebooks/notebooks.md)
- [Resources](resources/resources.md)
- [Scripts](scripts/scripts.md)
