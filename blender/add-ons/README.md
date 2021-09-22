# SMPL Blender Add-on

This add-on allows you to add gender specific [SMPL](https://smpl.is.tue.mpg.de) skinned meshes to your current Blender scene. Each imported SMPL mesh consist of a shape specific rig, as well as shape keys (blend shapes) for shape, expression and pose correctives.

Add-on features:
+ Add gender specific SMPL mesh to current scene
+ Set mesh texture
+ Position feet on ground plane (z=0)
+ Randomize/reset shape
+ Update joint locations
+ Enable/disable corrective poseshapes
+ Write current pose in SMPL theta notation to console
+ FBX export to Unity
    + Imported FBX will show up in Unity inspector without rotations and without scaling
    + Shape key export options: 
        + Body shape + posecorrectives
        + Body shape only
        + None (bakes current body shape into mesh)

Requirements: Blender 2.80+

Additional dependencies: None

## Installation
1. Blender>Edit>Preferences>Add-ons>Install
2. Select SMPL for Blender add-on ZIP file and install
3. Enable SMPL for Blender add-on
4. Enable sidebar in 3D Viewport>View>Sidebar
5. SMPL tool will show up in sidebar

## License
+ Generated body mesh data using this add-on:
    + Licensed under SMPL Model License
        + https://smpl.is.tue.mpg.de/modellicense

+ See LICENSE.md for further license information including commercial licensing

+ Attribution for publications: 
    + You agree to cite the most recent paper describing the model as specified on the SMPL website: https://smpl.is.tue.mpg.de
