# Character Animation in Blender
Done as part of a RnD project under Prof. Parag Chaudhari, IIT Bombay.

Most of the code has been cloned from [this](https://github.com/mkocabas/VIBE) repository.

# Directory structure

The Blender specific code is present in the folder _blender/_

* file_selector.py: Custom Blender plug-in for choosing a video file and extracting 3D human pose
* fbx_output.py: Converting the tracking output into an fbx file with animation
* add-ons/rokoko: Main add-on for retargetting the pose from one source armature to any target armature

# Installation

1. Install Blender from dmg
2. Alias python and pip to Blender’s python and installed pip
3. Install site-packages from requirements.txt
   (If can’t find OpenGL error, follow [this](https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos) link)
4. Open the script blenderfile_selector.py in Blender and run it to install the plugin in the object sidebar
5. Install Rokoko plugin from the zip file in _blender/add-ons_ folder
6. Choose gender
7. Choose a video file

