# Landmark-Recognition
A project to detect landmark from a query image and return all the images from the training set containing similar landmarks

# Technologies Used
TensorFlow Python

# Concepts
Deep Local Features (DeLF)

# Steps
Install tensorflow using virtual environment
Clone this repository
Commands : cd models/research/ 
export PYTHONPATH=$PYTHONPATH:pwd 
cd delf/delf/python/examples/ 
python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path data/small_data_radhu --output_dir data/small_data_radhu_features

