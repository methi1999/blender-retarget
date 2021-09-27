import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
from bpy.props import StringProperty
import os
import sys

# set paths
PWD = '/Users/mithileshvaidya/Code/HPR/VIBE'
gender_global = 'male'

blender_python_path = '/Applications/Blender.app/Contents/Resources/2.93/python/bin/python3.9'

sys.path.insert(0, PWD + '/blender')

# reload fxb_output because Blender does not reload imported libraries
# Check https://blender.stackexchange.com/questions/28504/blender-ignores-changes-to-python-scripts
import fbx_output
# When bpy is already in local, we know this is not the initial import...
if "bpy" in locals():
    # ...so we need to reload our submodule(s) using importlib
    import importlib
    if "fbx_output" in locals():
        importlib.reload(fbx_output)

global cur_arm_id

# info regarding blender plug-in
bl_info = {
    "name": "File selector",
    "author": "Mithilesh Vaidya",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Add > Mesh > New Object",
    "description": "Animate",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}


# file browser class
class OpenFilebrowser(Operator, ImportHelper):
    # idname is the link
    bl_idname = "test.open_filebrowser"
    bl_label = "Select input video file"
    # allow only mp4 files
    filter_glob: StringProperty(
        default='*.mp4',
        options={'HIDDEN'}
    )

    def execute(self, context):
        global cur_arm_id
        # gets called when file is chosen
        # path stored in self.filepath
        print("Input:", self.filepath)
        head, tail = os.path.split(self.filepath)
        tail, _ = os.path.splitext(tail)
        # call VIBE demo on input video
        vibe_pth = os.path.join(PWD, 'output', tail, 'vibe_output.pkl')
        if not os.path.exists(vibe_pth):
            print("Running VIBE since {} does not exist".format(vibe_pth))
            cmd = 'cd {}; {} demo.py --vid_file {} --output_folder output/'.format(PWD, blender_python_path, self.filepath)
            os.system(cmd)
        else:
            print("Found VIBE output at {}".format(vibe_pth))
        # once we have the pickle output, convert it to .fbx by calling demo.py
        fname = os.path.basename(self.filepath).replace('.mp4', '')
        # once we have the pickle output, convert it to .fbx by calling demo.py)
        print("Gender:", gender_global)
        fbx_output.pickle_to_fbx(cur_arm_id, os.path.join(PWD, 'output/{}/vibe_output.pkl'.format(fname)),
                                 os.path.join(PWD, 'output/{}/fbx_out.fbx'.format(fname)), gender_global, person_id=1)
        cur_arm_id += 1
        return {'FINISHED'}


# gender chooser class
class GenderChooser(bpy.types.Operator):
    bl_label = "Choose gender"
    bl_idname = "wm.template_operator"
    gender_enums: bpy.props.EnumProperty(items=[('male', 'Male', 'Choose Male SMPL model'),
                                                ('female', 'Female', 'Choose Female SMPL model')],
                                         name="Gender picker")

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "gender_enums")

    def execute(self, context):
        global gender_global
        gender_global = self.gender_enums
        print("Set gender to {}".format(gender_global))
        return {'FINISHED'}


# button layout panel
class LayoutDemoPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Layout Demo"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # Gender picker button
        layout.operator("wm.template_operator")
        # Input video button
        layout.label(text="Input video")
        row = layout.row()
        row.scale_y = 3.0
        row.operator("test.open_filebrowser")


classes = [OpenFilebrowser, LayoutDemoPanel, GenderChooser]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    cur_arm_id = 0
    register()
