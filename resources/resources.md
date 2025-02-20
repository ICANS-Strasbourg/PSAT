# Resources

## TCIA Dataset

## TotalSegmentator Dataset

## Checkpoints

We provide two model checkpoints that you can use directly with nnU-Net:

- **mixed_model_continual_learning.zip**  

- **pure_pediatric_model.zip**  

### Installing Pretrained Models

nnU-Net offers a convenient utility to install pretrained models from a zip archive. Follow these steps:

1. **Downloading Pretrained Weights:**  
   Go to the [GitHub Releases](https://github.com/ICANS-Strasbourg/PSAT/releases) page.
  
   Download:
   - `mixed_model_continual_learning.zip`
   - `pure_pediatric_model.zip`
 
   Place them in `resources/checkpoints/`.

2. **Install the Checkpoint Using nnU-Net**  
   Use the `nnUNet_install_pretrained_model_from_zip` command in your terminal.

   ```bash
   # To install the mixed model checkpoint:
   nnUNetv2_install_pretrained_model_from_zip resources/checkpoints/mixed_model_continual_learning.zip

   # To install the pure pediatric model checkpoint:
   nnUNetv2_install_pretrained_model_from_zip resources/checkpoints/pure_pediatric_model.zip

3. **Running Inference**

    Once you have installed a checkpoint, you can run inference on your input images using the `nnUNet_predict` command.
