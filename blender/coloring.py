import bpy
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from copy import deepcopy

np.random.seed(7)
random.seed(7)

bones1 = ['boss:Spine2', 'boss:Neck', 'boss:Neck1', 'boss:Head']
bones2 = ['m_avg_L_Wrist', 'm_avg_L_Hand', 'm_avg_R_Collar', 'm_avg_R_Shoulder']
assert len(bones1) == len(bones2)
arm1, arm2 = 'Armature', 'Armature_VIBE_0'

colors = [(random.random(), random.random(), random.random()) for _ in range(len(bones1))]
# color source
bpy.ops.armature.select_all(action='DESELECT')
bpy.data.objects[arm1].select_set(True)
bpy.ops.object.mode_set(mode = 'POSE')
for i, (s, t) in enumerate(zip(bones1, bones2)):
    color = colors[i]
    bpy.ops.pose.group_add()
    pose = bpy.data.objects[arm1].pose
    groups = pose.bone_groups
    index = len(groups) - 1

    groups[index].name = s + '::' + t
    groups[index].colors.active = color
    groups[index].colors.normal = color
    groups[index].colors.select = color
    groups[index].color_set = "CUSTOM"
    pose.bones[s].bone_group_index = index

# go back to object mode
bpy.ops.armature.select_all(action='DESELECT')

## colour target
bpy.data.objects[arm2].select_set(True)
bpy.ops.object.mode_set(mode = 'POSE')
for i, (s, t) in enumerate(zip(bones1, bones2)):
    color = colors[i]
    bpy.ops.pose.group_add()
    pose = bpy.data.objects[arm2].pose
    groups = pose.bone_groups
    index = len(groups) - 1

    groups[index].name = s + '::' + t
    groups[index].colors.active = color
    groups[index].colors.normal = color
    groups[index].colors.select = color
    groups[index].color_set = "CUSTOM"
    pose.bones[t].bone_group_index = index


    # delete
#    bpy.data.objects['Armature_VIBE_0'].pose.bone_groups.remove(
#        bpy.data.objects['Armature_VIBE_0'].pose.bone_groups['boss:Spine2::f_avg_L_Wrist'])
