# OIC-244 Merging histological images

The goal of this project is to merge two large histology images with Yue Ma (Moore Lab). The images are the same tissue samples which were stained with two different dyes.

The main challenge with this project is that the tissue becomes slightly deformed between the two images so a simple translation does not fully correct the shift. To solve this, the images are first coarsely aligned. Then the images are divided into sub-images to obtain finer alignment, which is then used to perform non-rigid registration. All alignments were calculated using phase correlation.

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) version 3.13.7 or higher

### Installation

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-244.git
   cd OIC-244
   ```

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   
   Linux:
   ```bash
   source venv/scripts/activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

2. > [!NOTE]
   > Non-rigid alignment is carried out using openCV's ``remap`` function. However, using this library restricts the image size to int16 size (i.e. 32767x32767). To avoid issues, larger images need to be cropped ahead of time.
   
   Crop images by running
   ```bash
   python -m export_region
   ```

   You will need to edit the file ``export_region.py`` and change the values of the variables ``imageA_path`` and ``imageB_path`` to point to the two files you want to use.  

3. Edit the script ``register_images.py`` and change the values of the variables ``target`` and ``moving`` to point to the images you want to merge. Then run

   ```bash
   python -m merge_data
   ```
   A combined TIFF-file will be created in a new directory ``processed``.

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-244/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Contributors
<a href="https://github.com/vaioic/OIC-244/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/OIC-244" />
</a>

### Dependencies

This project relies on the following packages:

- openCV v4.12

**Note:** For full dependency list, see [requirements.txt](requirements.txt).