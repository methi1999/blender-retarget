import os

import bpy
import random
import copy
from . import detector
from ..core import utils
from ..core.auto_detect_lists import bones
from ..core.retargeting import get_source_armature, get_target_armature
from ..core import detection_manager as detector
import sys
import os

PWD = '/Users/mithileshvaidya/Code/HPR/VIBE'
sys.path.insert(0, os.path.join(PWD, 'blender'))

import graph_match

if "bpy" in locals():
    # ...so we need to reload our submodule(s) using importlib
    import importlib

    if "graph_match" in locals():
        importlib.reload(graph_match)

RETARGET_ID = '_RSL_RETARGET'


class BuildBoneList(bpy.types.Operator):
    """
    Adds rows to the interface and maps bones according to retargeting dict
    """
    bl_idname = "rsl.build_bone_list"
    bl_label = "Build Bone List"
    bl_description = "Builds the bone list from the animation and tries to automatically detect and match bones"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        armature_source = get_source_armature()
        armature_target = get_target_armature()

        if not armature_source.animation_data or not armature_source.animation_data.action:
            self.report({'ERROR'}, 'No animation on the source armature found!'
                                   '\nSelect an armature with an animation as source.')
            return {'CANCELLED'}

        if armature_source.name == armature_target.name:
            self.report({'ERROR'}, 'Source and target armature are the same!'
                                   '\nPlease select different armatures.')
            return {'CANCELLED'}

        retargeting_dict = detector.detect_retarget_bones()

        # Clear the bone retargeting list
        context.scene.rsl_retargeting_bone_list.clear()

        for bone_source, bone_values in retargeting_dict.items():
            bone_target, bone_key = bone_values

            bone_item = context.scene.rsl_retargeting_bone_list.add()
            bone_item.bone_name_key = bone_key
            bone_item.bone_name_source = bone_source
            bone_item.bone_name_target = bone_target

        return {'FINISHED'}


# FirstGuess
class ExtractHierarchy(bpy.types.Operator):
    bl_label = "Extract hierarchy"
    bl_idname = "wm.extract_hierarchy"
    bl_options = {'REGISTER'}
    key_bones = []

    def draw(self, context):
        layout = self.layout
        layout.label(text="Extract Hierarchy")

    def modal(self, context, event):
        if event.type == 'ESC':
            # empty list
            ExtractHierarchy.key_bones = []
            print("Exiting hierarchy mode")
            return {'FINISHED'}
        elif event.type == 'P':
            if len(bpy.context.selected_pose_bones) != 1:
                print("Select exactly 1 bone at a time")
            else:
                bone = bpy.context.selected_pose_bones[-1].name
                # accidentally registers 3 presses
                if len(ExtractHierarchy.key_bones) == 0 or ExtractHierarchy.key_bones[-1] != bone:
                    ExtractHierarchy.key_bones.append(bone)
                    print(ExtractHierarchy.key_bones)
                # if 4 bones are selected, call function
                if len(ExtractHierarchy.key_bones) == 4:
                    source_arm, target_arm = get_source_armature(), get_target_armature()
                    source_root = bpy.data.armatures[source_arm.name].bones[
                        bpy.data.armatures[source_arm.name].bones[0].name]
                    target_root = bpy.data.armatures[target_arm.name].bones[
                        bpy.data.armatures[target_arm.name].bones[0].name]
                    print("Detected Source root: {}, Target root: {}".format(source_root.name, target_root.name))

                    def recursive_build(node, names):
                        names[node.name] = {}
                        for child in node.children.keys():
                            names[node.name][child] = recursive_build(node.children[child], names)
                        return names[node.name]

                    source_hier = {source_root.name: recursive_build(source_root, {})}
                    target_hier = {target_root.name: recursive_build(target_root, {})}
                    print("Source hierarchy", source_hier, '\n\n\n', 'Target hierarchy', target_hier)
                    # root split
                    best_mapping = graph_match.retarget_root_up(source_hier, target_hier, ExtractHierarchy.key_bones)
                    # arm/leg split
                    # best_mapping = graph_match.retarget_arms_legs(source_hier, target_hier, ExtractHierarchy.key_bones)
                    # set mapping
                    print("Best mapping:", best_mapping)
                    for bone_item in context.scene.rsl_retargeting_bone_list:
                        if bone_item.bone_name_source in best_mapping:
                            bone_item.bone_name_target = best_mapping[bone_item.bone_name_source]

                    # empty list
                    ExtractHierarchy.key_bones = []

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        print("Choose 4 bones for hierarchy: LA source, LA target, LL source, LL target")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):

        return {'FINISHED'}


# Colour
class ColourBones(bpy.types.Operator):
    bl_label = "Colour bones"
    bl_idname = "wm.colour_bones"
    bl_options = {'REGISTER'}
    bone_groups = []

    def draw(self, context):
        layout = self.layout
        layout.label(text="Colour bones")

    def modal(self, context, event):
        if event.type == 'ESC':
            # empty list
            ColourBones.bone_groups = []
            print("Exiting colour mode")
            return {'FINISHED'}
        elif event.type == 'P':
            if bpy.context.selected_pose_bones is not None:
                if len(bpy.context.selected_pose_bones) != 2:
                    print("Select start and end bones")
                else:
                    # get 2 bones
                    first, second = bpy.context.selected_pose_bones[-1].name, bpy.context.selected_pose_bones[-2].name
                    source_arm, target_arm = get_source_armature(), get_target_armature()
                    source_root = bpy.data.armatures[source_arm.name].bones[
                        bpy.data.armatures[source_arm.name].bones[0].name]
                    target_root = bpy.data.armatures[target_arm.name].bones[
                        bpy.data.armatures[target_arm.name].bones[0].name]
                    print("Detected Source root: {}, Target root: {}".format(source_root.name, target_root.name))

                    def recursive_build(node, names):
                        names[node.name] = {}
                        for child in node.children.keys():
                            names[node.name][child] = recursive_build(node.children[child], names)
                        return names[node.name]

                    source_hier = {source_root.name: recursive_build(source_root, {})}
                    bones_source = graph_match.get_nodes_in_path(source_hier, first, second)

                    retargeted_dict = {}
                    for bone_item in context.scene.rsl_retargeting_bone_list:
                        retargeted_dict[bone_item.bone_name_source] = bone_item.bone_name_target
                    bones_target = [retargeted_dict[x] for x in bones_source if
                                    retargeted_dict[x] is not None and len(retargeted_dict[x])]
                    # colour
                    colors = [(random.random(), random.random(), random.random()) for _ in range(len(bones_source))]
                    cur_names = []
                    # color source
                    for i, (s, t) in enumerate(zip(bones_source, bones_target)):
                        color = colors[i]
                        bpy.ops.pose.group_add()
                        pose = bpy.data.objects[source_arm.name].pose
                        groups = pose.bone_groups
                        index = len(groups) - 1

                        groups[index].name = s + '::' + t
                        cur_names.append(s + '::' + t)
                        groups[index].colors.active = color
                        groups[index].colors.normal = color
                        groups[index].colors.select = color
                        groups[index].color_set = "CUSTOM"
                        pose.bones[s].bone_group_index = index

                    ColourBones.bone_groups.append(cur_names)

                    # go back to object mode
                    bpy.ops.object.posemode_toggle()
                    bpy.data.objects[source_arm.name].hide_set(True)
                    # colour target
                    bpy.data.objects[target_arm.name].select_set(True)
                    bpy.context.view_layer.objects.active = bpy.data.objects[target_arm.name]
                    bpy.ops.object.posemode_toggle()
                    for i, (s, t) in enumerate(zip(bones_source, bones_target)):
                        color = colors[i]
                        bpy.ops.pose.group_add()
                        pose = bpy.data.objects[target_arm.name].pose
                        groups = pose.bone_groups
                        index = len(groups) - 1

                        groups[index].name = s + '::' + t
                        groups[index].colors.active = color
                        groups[index].colors.normal = color
                        groups[index].colors.select = color
                        groups[index].color_set = "CUSTOM"
                        pose.bones[t].bone_group_index = index

                    bpy.ops.object.posemode_toggle()
                    bpy.data.objects[source_arm.name].hide_set(False)
                    bpy.data.objects[target_arm.name].select_set(True)
                    bpy.ops.object.posemode_toggle()
        elif event.type == 'U':
            print("Removing colour groups")
            source_arm, target_arm = get_source_armature().name, get_target_armature().name
            for n in ColourBones.bone_groups[-1]:
                try:
                    bpy.data.objects[source_arm].pose.bone_groups.remove(bpy.data.objects[source_arm].pose.bone_groups[n])
                    bpy.data.objects[target_arm].pose.bone_groups.remove(bpy.data.objects[target_arm].pose.bone_groups[n])
                except:
                    print("Some error")
                    pass
            if len(ColourBones.bone_groups):
                del ColourBones.bone_groups[-1]

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        print("Choose 2 bones form source armature")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        return {'FINISHED'}


class ManualBoneMapper(bpy.types.Operator):
    """Process input while Control key is pressed."""
    bl_idname = 'custom.map_bones'
    bl_label = 'Enter manual mapping'
    bl_description = "On entering, choose source bone followed by target bone and press U. Exit mode using Esc"
    bl_options = {'REGISTER'}

    def modal(self, context, event):
        if event.type == 'ESC':
            print("Exiting mapping mode")
            return {'FINISHED'}
        elif event.type == 'U':
            if len(bpy.context.selected_pose_bones) != 2:
                print("Select exactly 2 bones to map")
            else:
                first, second = bpy.context.selected_pose_bones[-1].name, bpy.context.selected_pose_bones[-2].name
                found = False
                for bone_item in context.scene.rsl_retargeting_bone_list:
                    if bone_item.bone_name_source == first:
                        bone_item.bone_name_target = second
                        found = True
                        break
                    elif bone_item.bone_name_source == second:
                        bone_item.bone_name_target = first
                        found = True
                        break
                if found:
                    print("Mapped {} from source to {} in target".format(first, second))
                else:
                    print("Couldn't find {} in source".format(first))

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        print("Entered mapping mode")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class ClearBoneList(bpy.types.Operator):
    bl_idname = "rsl.clear_bone_list"
    bl_label = "Clear Bone List"
    bl_description = "Clears the bone list so that you can manually fill in all bones"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        for bone_item in context.scene.rsl_retargeting_bone_list:
            bone_item.bone_name_target = ''
        return {'FINISHED'}


class RetargetAnimation(bpy.types.Operator):
    bl_idname = "rsl.retarget_animation"
    bl_label = "Retarget Animation"
    bl_description = "Retargets the animation from the source armature to the target armature"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        armature_source = get_source_armature()
        armature_target = get_target_armature()

        if not armature_source.animation_data or not armature_source.animation_data.action:
            self.report({'ERROR'}, 'No animation on the source armature found!'
                                   '\nSelect an armature with an animation as source.')
            return {'CANCELLED'}

        if armature_source.name == armature_target.name:
            self.report({'ERROR'}, 'Source and target armature are the same!'
                                   '\nPlease select different armatures.')
            return {'CANCELLED'}

        # Find root bones
        root_bones = self.find_root_bones(context, armature_source, armature_target)

        # Cancel if no root bones are found
        if not root_bones:
            self.report({'ERROR'}, 'No root bone found!'
                                   '\nCheck if the bones are mapped correctly or try rebuilding the bone list.')
            return {'CANCELLED'}

        # Save the bone list if the user changed anything
        detector.save_retargeting_to_list()

        # Prepare armatures
        utils.set_active(armature_target)
        bpy.ops.object.mode_set(mode='OBJECT')
        utils.set_active(armature_source)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Set armatures into pose mode
        armature_source.data.pose_position = 'POSE'
        armature_target.data.pose_position = 'POSE'

        # Save and reset the current pose position of both armatures if rest position should be used
        pose_source, pose_target = {}, {}
        if bpy.context.scene.rsl_retargeting_use_pose == 'REST':
            pose_source = self.get_and_reset_pose_rotations(armature_source)
            pose_target = self.get_and_reset_pose_rotations(armature_target)

        source_scale = None
        if context.scene.rsl_retargeting_auto_scaling:
            # Clean source animation
            self.clean_animation(armature_source)

            # Scale the source armature to fit the target armature
            source_scale = copy.deepcopy(armature_source.scale)
            self.scale_armature(context, armature_source, armature_target, root_bones)

        # Duplicate source armature to apply transforms to the animation
        armature_source_original = armature_source
        armature_source = self.copy_rest_pose(context, armature_source)

        # Save transforms of target armature
        rotation_mode = armature_target.rotation_mode
        armature_target.rotation_mode = 'QUATERNION'
        rotation = copy.deepcopy(armature_target.rotation_quaternion)
        location = copy.deepcopy(armature_target.location)

        # Apply transforms of the target armature
        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature_target)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        bpy.ops.object.mode_set(mode='EDIT')

        # Create a transformation dict of all bones of the target armature and unselect all bones
        bone_transforms = {}
        for bone in context.object.data.edit_bones:
            bone.select = False
            bone_transforms[bone.name] = armature_source.matrix_world.inverted() @ bone.head.copy(), \
                                         armature_source.matrix_world.inverted() @ bone.tail.copy(), \
                                         utils.mat3_to_vec_roll(
                                             armature_source.matrix_world.inverted().to_3x3() @ bone.matrix.to_3x3())  # Head loc, tail loc, bone roll

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature_source)
        bpy.ops.object.mode_set(mode='EDIT')

        # Recreate bones from target armature in source armature
        for item in context.scene.rsl_retargeting_bone_list:
            if not item.bone_name_source or not item.bone_name_target or item.bone_name_target not in bone_transforms:
                continue

            bone_source = armature_source.data.edit_bones.get(item.bone_name_source)
            if not bone_source:
                print('Skipped:', item.bone_name_source, item.bone_name_target)
                continue

            # Recreate target bone
            bone_new = armature_source.data.edit_bones.new(item.bone_name_target + RETARGET_ID)
            bone_new.head, bone_new.tail, bone_new.roll = bone_transforms[item.bone_name_target]
            bone_new.parent = bone_source

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        # Add constraints to target armature and select the bones for animation
        for item in context.scene.rsl_retargeting_bone_list:
            if not item.bone_name_source or not item.bone_name_target:
                continue

            bone_source = armature_source.pose.bones.get(item.bone_name_source)
            bone_target = armature_target.pose.bones.get(item.bone_name_target)
            bone_target_data = armature_target.data.bones.get(item.bone_name_target)

            if not bone_source or not bone_target or not bone_target_data:
                print('Bone mapping not found:', item.bone_name_source, item.bone_name_target)
                continue

            # Add constraints
            constraint = bone_target.constraints.new('COPY_ROTATION')
            constraint.name += RETARGET_ID
            constraint.target = armature_source
            constraint.subtarget = item.bone_name_target + RETARGET_ID

            if bone_target.name in root_bones:
                constraint = bone_target.constraints.new('COPY_LOCATION')
                constraint.name += RETARGET_ID
                constraint.target = armature_source
                constraint.subtarget = item.bone_name_source

            # Select the bone for animation
            armature_target.data.bones.get(item.bone_name_target).select = True

        # Bake the animation to the target armature
        self.bake_animation(armature_source, armature_target, root_bones)

        # Delete the duplicate helper armature
        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature_source)
        bpy.data.actions.remove(armature_source.animation_data.action)
        bpy.ops.object.delete()

        # Change armature source back to original
        armature_source = armature_source_original

        # Change action name
        armature_target.animation_data.action.name = armature_source.animation_data.action.name + ' Retarget'

        # Remove constraints from target armature
        for bone in armature_target.pose.bones:
            for constraint in bone.constraints:
                if RETARGET_ID in constraint.name:
                    bone.constraints.remove(constraint)

        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature_target)

        # Reset target armature transforms to old state
        armature_target.rotation_quaternion = rotation
        armature_target.location = location

        armature_target.rotation_quaternion.w = -armature_target.rotation_quaternion.w
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        armature_target.rotation_quaternion = rotation
        armature_target.rotation_mode = rotation_mode

        # Reset source armature scale
        if source_scale:
            armature_source.scale = source_scale

        # Reset pose positions to old state
        # self.load_pose_rotations(armature_source, pose_source)
        # self.load_pose_rotations(armature_target, pose_target)

        bpy.ops.object.select_all(action='DESELECT')

        self.report({'INFO'}, 'Retargeted animation.')
        return {'FINISHED'}

    def find_root_bones(self, context, armature_source, armature_target):
        # Find all root bones
        root_bones = []
        for bone in armature_target.pose.bones:
            if not bone.parent:
                root_bones.append(bone)

        # Find animated root bones
        root_bones_animated = []
        target_bones = [item.bone_name_target for item in context.scene.rsl_retargeting_bone_list if
                        armature_target.pose.bones.get(item.bone_name_target) and armature_source.pose.bones.get(
                            item.bone_name_source)]
        while root_bones:
            for bone in copy.copy(root_bones):
                root_bones.remove(bone)
                if bone.name in target_bones:
                    root_bones_animated.append(bone.name)
                else:
                    for bone_child in bone.children:
                        root_bones.append(bone_child)
        return root_bones_animated

    def clean_animation(self, armature_source):
        deletable_fcurves = ['location', 'rotation_euler', 'rotation_quaternion', 'scale']
        for fcurve in armature_source.animation_data.action.fcurves:
            if fcurve.data_path in deletable_fcurves:
                armature_source.animation_data.action.fcurves.remove(fcurve)

    def get_and_reset_pose_rotations(self, armature):
        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature)
        bpy.ops.object.mode_set(mode='POSE')

        # Save rotations
        pose_rotations = {}
        for bone in armature.pose.bones:
            if bone.rotation_mode == 'QUATERNION':
                pose_rotations[bone.name] = copy.deepcopy(bone.rotation_quaternion)
                bone.rotation_quaternion = (1, 0, 0, 0)
            else:
                pose_rotations[bone.name] = copy.deepcopy(bone.rotation_euler)
                bone.rotation_euler = (0, 0, 0)

        # Reset rotations
        # bpy.ops.pose.rot_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

        return pose_rotations

    def load_pose_rotations(self, armature, pose_rotations):
        if not pose_rotations:
            return

        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature)
        bpy.ops.object.mode_set(mode='POSE')

        # Load rotations
        for bone in armature.pose.bones:
            rot = pose_rotations.get(bone.name)
            if rot:
                if bone.rotation_mode == 'QUATERNION':
                    bone.rotation_quaternion = rot
                else:
                    bone.rotation_euler = rot

        bpy.ops.object.mode_set(mode='OBJECT')

    def scale_armature(self, context, armature_source, armature_target, root_bones):
        source_min = None
        source_min_root = None
        target_min = None
        target_min_root = None

        for item in context.scene.rsl_retargeting_bone_list:
            if not item.bone_name_source or not item.bone_name_target:
                continue

            bone_source = armature_source.pose.bones.get(item.bone_name_source)
            bone_target = armature_target.pose.bones.get(item.bone_name_target)
            if not bone_source or not bone_target:
                continue

            bone_source_z = (armature_source.matrix_world @ bone_source.head)[2]
            bone_target_z = (armature_target.matrix_world @ bone_target.head)[2]

            if item.bone_name_target in root_bones:
                if source_min_root is None or source_min_root > bone_source_z:
                    source_min_root = bone_source_z
                if target_min_root is None or target_min_root > bone_target_z:
                    target_min_root = bone_target_z

            if source_min is None or source_min > bone_source_z:
                source_min = bone_source_z
            if target_min is None or target_min > bone_target_z:
                target_min = bone_target_z

        source_height = source_min_root - source_min
        target_height = target_min_root - target_min

        if not source_height or not target_height:
            print('No scaling needed')
            return

        scale_factor = target_height / source_height
        armature_source.scale *= scale_factor

    def read_anim_start_end(self, armature):
        frame_start = None
        frame_end = None
        for fcurve in armature.animation_data.action.fcurves:
            for key in fcurve.keyframe_points:
                keyframe = key.co.x
                if frame_start is None:
                    frame_start = keyframe
                if frame_end is None:
                    frame_end = keyframe

                if keyframe < frame_start:
                    frame_start = keyframe
                if keyframe > frame_end:
                    frame_end = keyframe

        return frame_start, frame_end

    def copy_rest_pose(self, context, armature_source):
        # make sure auto keyframe is disabled, leads to issues
        context.scene.tool_settings.use_keyframe_insert_auto = False

        # ensure the source armature selection
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(armature_source)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Duplicate the source armature
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked": False, "mode": 'TRANSLATION'},
                                      TRANSFORM_OT_translate={"value": (0, 0, 0),
                                                              "constraint_axis": (False, True, False), "mirror": False,
                                                              "snap": False, "remove_on_cancel": False,
                                                              "release_confirm": False})

        # Set name of the copied source armature
        source_armature_copy = context.object
        source_armature_copy.name = armature_source.name + "_copy"

        bpy.ops.object.select_all(action='DESELECT')
        utils.set_active(source_armature_copy)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='POSE')

        # Apply transforms of the new source armature. Unlink action temporarily to prevent warning in console
        action_tmp = source_armature_copy.animation_data.action
        source_armature_copy.animation_data.action = None
        bpy.ops.pose.armature_apply()
        source_armature_copy.animation_data.action = action_tmp

        # Mimic the animation of the original source armature by adding constraints to the bones.
        # -> the new armature has the exact same animation but with applied transforms
        for bone in source_armature_copy.pose.bones:
            constraint = bone.constraints.new('COPY_TRANSFORMS')
            constraint.name = bone.name
            constraint.target = armature_source
            constraint.subtarget = bone.name

        bpy.ops.object.mode_set(mode='OBJECT')

        return source_armature_copy

    def bake_animation(self, armature_source, armature_target, root_bones):
        frame_split = 25
        frame_start, frame_end = self.read_anim_start_end(armature_source)
        frame_start, frame_end = int(frame_start), int(frame_end)
        utils.set_active(armature_target)

        actions_all = []

        # Setup loading bar
        current_step = 0
        steps = int((frame_end - frame_start) / frame_split) + 1
        wm = bpy.context.window_manager
        wm.progress_begin(current_step, steps)

        import time
        start_time = time.time()

        # Bake the animation in parts because multiple short parts are processed much faster than one long animation
        for frame in range(frame_start, frame_end + 2, frame_split):
            start = frame
            end = frame + frame_split - 1
            if end > frame_end:
                end = frame_end
            if start > end:
                continue

            # Bake animation part
            bpy.ops.nla.bake(frame_start=start, frame_end=end, visual_keying=True, only_selected=True,
                             use_current_action=False, bake_types={'POSE'})

            # Rename animation part
            armature_target.animation_data.action.name = 'RSL_RETARGETING_' + str(frame)

            actions_all.append(armature_target.animation_data.action)

            current_step += 1
            if steps != current_step:
                wm.progress_update(current_step)

        if not actions_all:
            return

        # Count all keys for all data_paths
        key_counts = {}
        for action in actions_all:
            for fcurve in action.fcurves:
                key = fcurve.data_path + str(fcurve.array_index)
                if not key_counts.get(key):
                    key_counts[key] = 0
                key_counts[key] += len(fcurve.keyframe_points)

        # Create new action
        action_final = bpy.data.actions.new(name='RSL_RETARGETING_FINAL')
        action_final.use_fake_user = True
        armature_target.animation_data_create().action = action_final

        # Put all baked animations parts back together into one
        print_i = 0
        for fcurve in actions_all[0].fcurves:
            if fcurve.data_path.endswith('scale'):
                continue
            if fcurve.data_path.endswith('location'):
                bone_name = fcurve.data_path.split('"')
                if len(bone_name) != 3:
                    continue
                if bone_name[1] not in root_bones:
                    continue

            curve_final = action_final.fcurves.new(data_path=fcurve.data_path, index=fcurve.array_index,
                                                   action_group=fcurve.group.name)
            keyframe_points = curve_final.keyframe_points
            keyframe_points.add(key_counts[fcurve.data_path + str(fcurve.array_index)])

            index = 0
            for action in actions_all:
                fcruve_to_add = action.fcurves.find(data_path=fcurve.data_path, index=fcurve.array_index)

                for kp in fcruve_to_add.keyframe_points:
                    keyframe_points[index].co.x = kp.co.x
                    keyframe_points[index].co.y = kp.co.y
                    keyframe_points[index].interpolation = 'LINEAR'
                    index += 1

            print_i += 1

        # Clean up animation. Delete all keyframes the use the same value as the previous and next one
        for fcurve in action_final.fcurves:
            if len(fcurve.keyframe_points) <= 2:
                continue

            kp_pre_pre = fcurve.keyframe_points[0]
            kp_pre = fcurve.keyframe_points[1]

            kp_to_delete = []
            for kp in fcurve.keyframe_points[2:]:
                if round(kp_pre_pre.co.y, 5) == round(kp_pre.co.y, 5) == round(kp.co.y, 5):
                    kp_to_delete.append(kp_pre)
                kp_pre_pre = kp_pre
                kp_pre = kp

            for kp in reversed(kp_to_delete):
                fcurve.keyframe_points.remove(kp)

        # Delete all baked animation parts, only the combined one is needed
        for action in actions_all:
            bpy.data.actions.remove(action)

        print('Retargeting Time:', round(time.time() - start_time, 2), 'seconds')
        wm.progress_end()
