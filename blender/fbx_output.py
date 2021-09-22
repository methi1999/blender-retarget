# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#
# Author: Joachim Tesch, Max Planck Institute for Intelligent Systems, Perceiving Systems
#
# Create keyframed animated skinned SMPL mesh from .pkl pose description
#
# Generated mesh will be exported in FBX or glTF format
#
# Notes:
#  + Male and female gender models only
#  + Script can be run from command line or in Blender Editor (Text Editor>Run Script)
#  + Command line: Install mathutils module in your bpy virtualenv with 'pip install mathutils==2.81.2'

import os
import sys
import time
from math import radians
import pickle

import bpy
import joblib
import numpy as np
from mathutils import Matrix, Vector, Quaternion

PWD = '/Users/mithileshvaidya/Code/HPR/VIBE/'

# Globals
male_model_path = os.path.join(PWD,
                               'data/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx')
female_model_path = os.path.join(PWD,
                                 'data/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx')

fps_source = 30
fps_target = 30

gender = 'male'

start_origin = 1

bone_name_from_index = {
    0: 'Pelvis',
    1: 'L_Hip',
    2: 'R_Hip',
    3: 'Spine1',
    4: 'L_Knee',
    5: 'R_Knee',
    6: 'Spine2',
    7: 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}


def get_vibe_arm_name(arm_id):
    return 'Armature_VIBE_{}'.format(str(arm_id))


# Helper functions

# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
# Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


# Setup scene
def setup_scene(arm_id, model_path, fps_target):
    scene = bpy.data.scenes['Scene']

    ###########################
    # Engine independent setup
    ###########################

    scene.render.fps = fps_target

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # ensure armature Armature doesnt already exist
    arm_name = get_vibe_arm_name(arm_id)
    if arm_name in bpy.data.armatures.keys():
        raise Exception("Key {} already in armatures".format(arm_name))
    elif 'Armature' in bpy.data.armatures.keys():
        raise Exception("Please rename 'Armature' to avoid name conflicts")

    # ensure object Armature doesnt already exist
    arm_name = get_vibe_arm_name(arm_id)
    if arm_name in bpy.data.objects.keys():
        raise Exception("Key {} already in objects".format(arm_name))
    elif 'Armature' in bpy.data.objects.keys():
        raise Exception("Please rename the object 'Armature' to avoid name conflicts")

    # Import gender specific .fbx template file
    # this will include naming scheme of armature
    bpy.ops.import_scene.fbx(filepath=model_path)
    # rename 'Armature' to 'Armature_armid
    bpy.data.armatures['Armature'].name = arm_name
    bpy.data.objects['Armature'].name = arm_name


# Process single pose into keyframed bone orientations
def process_pose(arm_id, current_frame, pose, trans, pelvis_position):
    if pose.shape[0] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)

    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]

    # Set the location of the Pelvis bone to the translation parameter
    arm_name = get_vibe_arm_name(arm_id)
    armature = bpy.data.objects[arm_name]
    bones = armature.pose.bones

    # Pelvis: X-Right, Y-Up, Z-Forward (Blender -Y)

    # Set absolute pelvis location relative to Pelvis bone head
    bones[bone_name_from_index[0]].location = Vector((100 * trans[1], 100 * trans[2], 100 * trans[0])) - pelvis_position

    # bones['Root'].location = Vector(trans)
    bones[bone_name_from_index[0]].keyframe_insert('location', frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone = bones[bone_name_from_index[index]]

        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

        if index == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
            bone.rotation_quaternion = bone_rotation

        bone.keyframe_insert('rotation_quaternion', frame=current_frame)

    return


# Process all the poses from the pose file
def process_poses(arm_id, input_path, gender, fps_source, fps_target, start_origin, person_id=1):
    print('Processing: ' + input_path)

    data = joblib.load(input_path)
    poses = data[person_id]['pose']
    trans = np.zeros((poses.shape[0], 3))
    flength = poses.shape[0]
    # ToDO: feed correct img_size. Discuss camera
    img_size = 1280
    orig_cam = data[1]['orig_cam']

    for i, cam in enumerate(orig_cam):
        cam_s = cam[0:1]
        cam_pos = cam[2:]
        tz = flength / (0.5 * img_size * cam_s)
        trans[i] = np.hstack([cam_pos, tz])

    if gender == 'female':
        model_path = female_model_path
        for k, v in bone_name_from_index.items():
            if '_avg_' not in v[1:]:
                bone_name_from_index[k] = 'f_avg_' + v
    elif gender == 'male':
        model_path = male_model_path
        for k, v in bone_name_from_index.items():
            if '_avg_' not in v[1:]:
                bone_name_from_index[k] = 'm_avg_' + v
    else:
        print('ERROR: Unsupported gender: ' + gender)
        sys.exit(1)

    # Limit target fps to source fps
    if fps_target > fps_source:
        fps_target = fps_source

    print(f'Gender: {gender}')
    print(f'Number of source poses: {str(poses.shape[0])}')
    print(f'Source frames-per-second: {str(fps_source)}')
    print(f'Target frames-per-second: {str(fps_target)}')
    print('--------------------------------------------------')

    setup_scene(arm_id, model_path, fps_target)

    scene = bpy.data.scenes['Scene']
    sample_rate = int(fps_source / fps_target)
    scene.frame_end = (int)(poses.shape[0] / sample_rate)

    # Retrieve pelvis world position.
    # Unit is [cm] due to Armature scaling.
    # Need to make copy since reference will change when bone location is modified.
    bpy.ops.object.mode_set(mode='EDIT')
    pelvis_position = Vector(bpy.data.armatures[get_vibe_arm_name(arm_id)].edit_bones[bone_name_from_index[0]].head)
    bpy.ops.object.mode_set(mode='OBJECT')
    source_index = 0
    frame = 1

    offset = np.array([0.0, 0.0, 0.0])

    while source_index < poses.shape[0]:
        print('Adding pose: ' + str(source_index))

        if start_origin:
            if source_index == 0:
                # so that the character start at the origin
                offset = np.array([trans[source_index][0], trans[source_index][1], trans[source_index][2]])

        # Go to new frame
        scene.frame_set(frame)
        process_pose(arm_id, frame, poses[source_index], (trans[source_index] - offset), pelvis_position)
        source_index += sample_rate
        frame += 1

    return frame


def export_animated_mesh(arm_id, output_path):
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Select only skinned mesh and rig
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[get_vibe_arm_name(arm_id)].select_set(True)
    bpy.data.objects[get_vibe_arm_name(arm_id)].children[0].select_set(True)

    if output_path.endswith('.glb'):
        print('Exporting to glTF binary (.glb)')
        # Currently exporting without shape/pose shapes for smaller file sizes
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_selected=True, export_morph=False)
    elif output_path.endswith('.fbx'):
        print('Exporting to FBX binary (.fbx)')
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False)
    else:
        print('ERROR: Unsupported export format: ' + output_path)
        sys.exit(1)

    return


def pickle_to_fbx(arm_id, input_path, output_path, gender, person_id=1):
    if not os.path.exists(input_path):
        print('ERROR: Invalid input path')
        return

    startTime = time.perf_counter()

    # Process data
    cwd = os.getcwd()

    # Turn relative input/output paths into absolute paths
    if not input_path.startswith(os.path.sep):
        input_path = os.path.join(cwd, input_path)

    if not output_path.startswith(os.path.sep):
        output_path = os.path.join(cwd, output_path)

    print('Input path: ' + input_path)
    print('Output path: ' + output_path)

    if not (output_path.endswith('.fbx') or output_path.endswith('.glb')):
        print('ERROR: Invalid output format (must be .fbx or .glb)')
        sys.exit(1)

    # Process pose file
    if input_path.endswith('.pkl'):
        if not os.path.isfile(input_path):
            print('ERROR: Invalid input file')
            sys.exit(1)

        poses_processed = process_poses(
            arm_id,
            input_path=input_path,
            gender=gender,
            fps_source=fps_source,
            fps_target=fps_target,
            start_origin=start_origin,
            person_id=person_id
        )
        export_animated_mesh(arm_id, output_path)

        print('--------------------------------------------------')
        print('Animation export finished.')
        print(f'Poses processed: {str(poses_processed)}')
        print(f'Processing time : {time.perf_counter() - startTime:.2f} s')
        print('--------------------------------------------------')
    else:
        raise Exception("Cant read pickle")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str)

    parser.add_argument('--output_path', type=str, default='output/',
                        help='output folder to write results')

    parser.add_argument('--gender', type=str, default='male')

    args = parser.parse_args()

    pickle_to_fbx(args.input_path, args.output_path, args.gender)