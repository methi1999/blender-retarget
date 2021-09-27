import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from copy import deepcopy

np.random.seed(7)


def graph_from_dict(nested_dict, undirected=False):
    """
    Convert nested dictionary to a networkx graph object
    @param nested_dict: {b1: {b2: {}, b3:{}} -> simple 3 node graph
    @param undirected: whether graph is undirected
    @return:
    """
    # Empty directed graph
    G = nx.DiGraph()

    # Iterate through the layers
    q = list(nested_dict.items())
    while q:
        v, d = q.pop()
        for nv, nd in d.items():
            G.add_edge(v, nv)
            if undirected:
                G.add_edge(nv, v)
            if isinstance(nd, dict):
                q.append((nv, nd))

    return G


def get_sub_dict(key, entire_d):
    """
    Extract subparts of skeleton hierarchy
    @param key: key to extract
    @param entire_d: original dictionary
    @return: if d = {a: {b: {c}, d: {}}} and key = b, return {b: {c}}
    """
    q = list(entire_d.items())
    while q:
        v, d = q.pop()
        if v == key:
            return {v: d}
        for nv, nd in d.items():
            if isinstance(nd, dict):
                q.append((nv, nd))


def delete_sub_dict(key, entire_d):
    """
    Delete a sub dictionary of key = key from entire dictionary
    @param key: to delete
    @param entire_d: original dictionary
    @return: None, in place deletion
    """
    if key in entire_d:
        del entire_d[key]
    for value in entire_d.values():
        if isinstance(value, dict):
            delete_sub_dict(key, value)


def get_nodes_in_path(d, start, end):
    """
    Get list of nodes in path; mianly used for coloring the bones
    @param d: nested dictionary for constructing graph
    @param start: start node
    @param end: end node
    @return: list of nodes [start, n1, n2, ..., end]
    """
    G = graph_from_dict(d, undirected=True)
    pth = list(nx.all_simple_paths(G, source=start, target=end))
    if len(pth) == 1:
        return pth[0]
    elif len(pth) == 0:
        # try reverse path since graph could be directed
        pth = list(nx.all_simple_paths(G, source=end, target=start))
        if len(pth) == 0:
            raise Exception("No path exists")
        else:
            return pth[0]


def best_matching(s, t):
    """
    Main function which generates best guess
    @param s: source dict
    @param t: target dict
    @return: {source_bone_1: best_target_guess_1, ... }
    """
    # build graphs
    source_g, target_g = graph_from_dict(s), graph_from_dict(t)
    # draw_graph([source_g, target_g])
    # transform target (G1) to source (G2)
    paths = nx.optimize_edit_paths(source_g, target_g, timeout=120)
    # find min cost result since paths will be a list of predictions
    min_nodes, min_cost = None, np.inf
    for n, e, c in paths:
        if c < min_cost:
            min_cost = c
            min_nodes = n
    return min_nodes


def retarget_arms_legs(source_dict, target_dict, arm_leg_mapping=None):
    # best first guess
    best_match = best_matching(source_dict, target_dict)
    final = {}
    for sbone, tbone in best_match:
        if sbone is not None and tbone is not None:
            final[sbone] = tbone

    print("Preliminary best map:", final)
    # separate for arms and legs
    if arm_leg_mapping is not None:
        # replace left-right
        pairs = [('_l_', '_r_'), ('left', 'right'), ('_L_', '_R_'), ('Left', 'Right')]
        for i in range(len(arm_leg_mapping)):
            source_b = arm_leg_mapping[i]
            for l, r in pairs:
                if l in source_b:
                    flipped = source_b.replace(l, r)
                    arm_leg_mapping[i] = (source_b, flipped)
                    break
                elif r in source_b:
                    flipped = source_b.replace(r, l)
                    arm_leg_mapping[i] = (source_b, flipped)
                    break
        """
        Order of tuples in arm_leg_mapping:
        (left arm source, right arm source), (left arm target, right arm target)
        (left leg source, right leg source), (left leg target, right leg target)
        """
        la = best_matching(get_sub_dict(arm_leg_mapping[0][0], source_dict),
                           get_sub_dict(arm_leg_mapping[1][0], target_dict))
        ra = best_matching(get_sub_dict(arm_leg_mapping[0][1], source_dict),
                           get_sub_dict(arm_leg_mapping[1][1], target_dict))
        ll = best_matching(get_sub_dict(arm_leg_mapping[2][0], source_dict),
                           get_sub_dict(arm_leg_mapping[3][0], target_dict))
        rl = best_matching(get_sub_dict(arm_leg_mapping[2][1], source_dict),
                           get_sub_dict(arm_leg_mapping[3][1], target_dict))

        for s, t in la:
            if s is not None and t is not None:
                final[s] = t
        for s, t in ra:
            if s is not None and t is not None:
                final[s] = t
        for s, t in ll:
            if s is not None and t is not None:
                final[s] = t
        for s, t in rl:
            if s is not None and t is not None:
                final[s] = t

    return resolve_left_right(final)


def resolve_left_right(d):
    """
    Resolves left-right errors after matching
    e.g. if 'L.Wrist' is mapped to 'right_Hand1', return {'L.Wrist': 'left_Hand1'}
    @param d: best mapping
    @return: resolved
    """
    l_r = {'_l': '_r', '.l': '.r', '_L': '_R', '.L': '.R', 'left': 'right', 'Left': 'Right'}
    for source, target in d.items():
        is_left, is_right = False, False
        for source_s, target_s in l_r.items():
            if source_s in source:
                is_left = True
                break
            elif target_s in source:
                is_right = True
                break
        if is_left:
            # make target left
            for l, r in l_r.items():
                if r in target:
                    flipped = target.replace(r, l)
                    print("Replacing {} with {}".format(d[source], flipped))
                    d[source] = flipped
                    break
        elif is_right:
            # make target right
            for l, r in l_r.items():
                if l in target:
                    flipped = target.replace(l, r)
                    print("Replacing {} with {}".format(d[source], flipped))
                    d[source] = flipped
                    break

    return d


def retarget_root_up(source_dict, target_dict, root_up):
    """
    Given root and up bones for source and target armature, split the graph and run edit distance matching
    @param source_dict:
    @param target_dict:
    @param root_up: [source root, source up, target root, target up]
    @return: best matching dict: {source_bone_1: best_target_bone1, ... }
    """
    source_root_name, source_up_name, target_root_name, target_up_name = root_up
    # make source up and down dicts
    source_up = {source_up_name: get_sub_dict(source_up_name, source_dict)}
    source_down = deepcopy(source_dict)
    delete_sub_dict(source_up_name, source_down)
    # for VIBE, the root is static. Dont consider it
    if 'f_avg_root' in source_down:
        source_down = source_down['f_avg_root']
    elif 'm_avg_root' in source_down:
        source_down = source_down['m_avg_root']
    # make target up and down dicts
    target_up = {target_up_name: get_sub_dict(target_up_name, target_dict)}
    target_down = deepcopy(target_dict)
    delete_sub_dict(target_up_name, target_down)
    if 'f_avg_root' in target_down:
        target_down = target_down['f_avg_root']
    elif 'm_avg_root' in target_down:
        target_down = target_down['m_avg_root']
    # run graph matching separately
    up_match = best_matching(source_up, target_up)
    down_match = best_matching(source_down, target_down)
    final = {}
    # get best mapping
    for sbone, tbone in up_match + down_match:
        if sbone is not None and tbone is not None:
            final[sbone] = tbone

    return resolve_left_right(final)


def iterative(s, t):
    """
    Iterative procedure from paper "Automatically Mapping Human Skeletons onto Virtual Character Armatures by Sanna"
    Very poor results
    @param s: source dict
    @param t: target dict
    @return: best matching
    """
    g1, g2 = graph_from_dict(s), graph_from_dict(t)
    # draw_graph(g1)
    g1_nodes, g2_nodes = list(g1.nodes()), list(g2.nodes())
    n_a, n_b, e_a, e_b = len(g1_nodes), len(g2_nodes), len(g1.edges), len(g2.edges)
    # construct source and terminus graphs
    a1 = nx.linalg.graphmatrix.incidence_matrix(g1, nodelist=g1_nodes, oriented=True).toarray()
    a_s, a_t = (a1 == 1).astype(np.uint8), (a1 == -1).astype(np.uint8)
    a2 = nx.linalg.graphmatrix.incidence_matrix(g2, nodelist=g2_nodes, oriented=True).toarray()
    b_s, b_t = (a2 == 1).astype(np.uint8), (a2 == -1).astype(np.uint8)
    # iterative updates
    # n_scores = np.ones((n_b, n_a))
    n_scores = np.random.random((n_b, n_a))
    n_scores /= np.linalg.norm(n_scores)
    e_scores = (np.transpose(np.kron(a_s, b_s) + np.kron(a_t, b_t)) @ n_scores.flatten('F')[:, None]).reshape((e_b, e_a), order='C')
    e_scores /= np.linalg.norm(e_scores)
    iters = 300
    for _ in range(iters):
        new_e = np.transpose(b_s)@n_scores@a_s + np.transpose(b_t)@n_scores@a_t
        new_n = b_s@e_scores@np.transpose(a_s) + b_t@e_scores@np.transpose(a_t)
        n_scores, e_scores = new_n/np.linalg.norm(new_n), new_e/np.linalg.norm(new_e)
    # print(n_scores, e_scores)
    best_nodes = np.argmax(n_scores, axis=0)
    mapping = {}
    for i, n in enumerate(g1_nodes):
        # print(best_nodes[i], n)
        mapping[n] = g2_nodes[best_nodes[i]]
    print(mapping)


def draw_graph(G_list):
    np.random.seed(8)
    n = len(G_list)
    for i, g in enumerate(G_list):
        plt.subplot(1, n, i+1)
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, with_labels=False)#node_color=np.random.rand(3)
        # nx.draw(g, with_labels=False)
    plt.show()


if __name__ == '__main__':
    # a_upper = {'f_avg_Spine1': {'f_avg_Spine2': {'f_avg_Spine3': {'f_avg_Neck': {'f_avg_Head': {}}, 'f_avg_L_Collar': {'f_avg_L_Shoulder': {'f_avg_L_Elbow': {'f_avg_L_Wrist': {'f_avg_L_Hand': {}}}}}, 'f_avg_R_Collar': {'f_avg_R_Shoulder': {'f_avg_R_Elbow': {'f_avg_R_Wrist': {'f_avg_R_Hand': {}}}}}}}}}
    # b_upper = {'boss:Spine': {'boss:Spine1': {'boss:Spine2': {'boss:Neck': {'boss:Neck1': {'boss:Head': {}}}, 'boss:LeftShoulder': {'boss:LeftArm': {'boss:LeftForeArm': {'boss:LeftHand': {'boss:LeftHandThumb1': {'boss:LeftHandThumb2': {'boss:LeftHandThumb3': {'boss:LeftHandThumb4': {}}}}, 'boss:LeftHandMiddle1': {'boss:LeftHandMiddle2': {'boss:LeftHandMiddle3': {'boss:LeftHandMiddle4': {}}}}, 'boss:LeftHandIndex1': {'boss:LeftHandIndex2': {'boss:LeftHandIndex3': {'boss:LeftHandIndex4': {}}}}, 'boss:LeftHandRing1': {'boss:LeftHandRing2': {'boss:LeftHandRing3': {'boss:LeftHandRing4': {}}}}, 'boss:LeftHandPinky1': {'boss:LeftHandPinky2': {'boss:LeftHandPinky3': {'boss:LeftHandPinky4': {}}}}}}}}, 'boss:RightShoulder': {'boss:RightArm': {'boss:RightForeArm': {'boss:RightHand': {'boss:RightHandThumb1': {'boss:RightHandThumb2': {'boss:RightHandThumb3': {'boss:RightHandThumb4': {}}}}, 'boss:RightHandIndex1': {'boss:RightHandIndex2': {'boss:RightHandIndex3': {'boss:RightHandIndex4': {}}}}, 'boss:RightHandMiddle1': {'boss:RightHandMiddle2': {'boss:RightHandMiddle3': {'boss:RightHandMiddle4': {}}}}, 'boss:RightHandRing1': {'boss:RightHandRing2': {'boss:RightHandRing3': {'boss:RightHandRing4': {}}}}, 'boss:RightHandPinky1': {'boss:RightHandPinky2': {'boss:RightHandPinky3': {'boss:RightHandPinky4': {}}}}}}}}}}}}
    # a_lower = {'f_avg_Pelvis': {'f_avg_L_Hip': {'f_avg_L_Knee': {'f_avg_L_Ankle': {'f_avg_L_Foot': {}}}}, 'f_avg_R_Hip': {'f_avg_R_Knee': {'f_avg_R_Ankle': {'f_avg_R_Foot': {}}}}}}
    # b_lower = {'boss:Hips': {'boss:LeftUpLeg': {'boss:LeftLeg': {'boss:LeftFoot': {'boss:LeftToeBase': {}}}}, 'boss:RightUpLeg': {'boss:RightLeg': {'boss:RightFoot': {'boss:RightToeBase': {}}}}}}
    a = {'f_avg_root': {'f_avg_Pelvis': {'f_avg_L_Hip': {'f_avg_L_Knee': {'f_avg_L_Ankle': {'f_avg_L_Foot': {}}}},
                                         'f_avg_R_Hip': {'f_avg_R_Knee': {'f_avg_R_Ankle': {'f_avg_R_Foot': {}}}},
                                         'f_avg_Spine1': {'f_avg_Spine2': {
                                             'f_avg_Spine3': {'f_avg_Neck': {'f_avg_Head': {}}, 'f_avg_L_Collar': {
                                                 'f_avg_L_Shoulder': {
                                                     'f_avg_L_Elbow': {'f_avg_L_Wrist': {'f_avg_L_Hand': {}}}}},
                                                              'f_avg_R_Collar': {'f_avg_R_Shoulder': {'f_avg_R_Elbow': {
                                                                  'f_avg_R_Wrist': {'f_avg_R_Hand': {}}}}}}}}}}}
    b = {'boss:Hips': {'boss:Spine': {'boss:Spine1': {'boss:Spine2': {'boss:Neck': {'boss:Neck1': {'boss:Head': {}}},
                                                                      'boss:LeftShoulder': {'boss:LeftArm': {
                                                                          'boss:LeftForeArm': {'boss:LeftHand': {
                                                                              'boss:LeftHandThumb1': {
                                                                                  'boss:LeftHandThumb2': {
                                                                                      'boss:LeftHandThumb3': {
                                                                                          'boss:LeftHandThumb4': {}}}},
                                                                              'boss:LeftHandMiddle1': {
                                                                                  'boss:LeftHandMiddle2': {
                                                                                      'boss:LeftHandMiddle3': {
                                                                                          'boss:LeftHandMiddle4': {}}}},
                                                                              'boss:LeftHandIndex1': {
                                                                                  'boss:LeftHandIndex2': {
                                                                                      'boss:LeftHandIndex3': {
                                                                                          'boss:LeftHandIndex4': {}}}},
                                                                              'boss:LeftHandRing1': {
                                                                                  'boss:LeftHandRing2': {
                                                                                      'boss:LeftHandRing3': {
                                                                                          'boss:LeftHandRing4': {}}}},
                                                                              'boss:LeftHandPinky1': {
                                                                                  'boss:LeftHandPinky2': {
                                                                                      'boss:LeftHandPinky3': {
                                                                                          'boss:LeftHandPinky4': {}}}}}}}},
                                                                      'boss:RightShoulder': {'boss:RightArm': {
                                                                          'boss:RightForeArm': {'boss:RightHand': {
                                                                              'boss:RightHandThumb1': {
                                                                                  'boss:RightHandThumb2': {
                                                                                      'boss:RightHandThumb3': {
                                                                                          'boss:RightHandThumb4': {}}}},
                                                                              'boss:RightHandIndex1': {
                                                                                  'boss:RightHandIndex2': {
                                                                                      'boss:RightHandIndex3': {
                                                                                          'boss:RightHandIndex4': {}}}},
                                                                              'boss:RightHandMiddle1': {
                                                                                  'boss:RightHandMiddle2': {
                                                                                      'boss:RightHandMiddle3': {
                                                                                          'boss:RightHandMiddle4': {}}}},
                                                                              'boss:RightHandRing1': {
                                                                                  'boss:RightHandRing2': {
                                                                                      'boss:RightHandRing3': {
                                                                                          'boss:RightHandRing4': {}}}},
                                                                              'boss:RightHandPinky1': {
                                                                                  'boss:RightHandPinky2': {
                                                                                      'boss:RightHandPinky3': {
                                                                                          'boss:RightHandPinky4': {}}}}}}}}}}},
                       'boss:LeftUpLeg': {'boss:LeftLeg': {'boss:LeftFoot': {'boss:LeftToeBase': {}}}},
                       'boss:RightUpLeg': {'boss:RightLeg': {'boss:RightFoot': {'boss:RightToeBase': {}}}}}}
    a_l_arm, a_r_arm = 'f_avg_L_Shoulder', 'f_avg_R_Shoulder'
    a_l_leg, a_r_leg = 'f_avg_L_Knee', 'f_avg_R_Knee'
    b_l_arm, b_r_arm = 'boss:LeftArm', 'boss:RightArm'
    b_l_leg, b_r_leg = 'boss:LeftLeg', 'boss:RightLeg'

    get_nodes_in_path(a, 'f_avg_R_Hip', 'f_avg_L_Wrist')

    # arms testing
    # retarget(get_sub_dict(a_l_arm, a), get_sub_dict(b_l_arm, b))
    # retarget(get_sub_dict(a_l_leg, a), get_sub_dict(b_l_leg, b))
    # retarget_arms_legs(a_lower, b_lower)
    # retarget_arms_legs(a_upper, b_upper)

    # root testing
    # retarget_root_up(a, b, ['f_avg_Pelvis', 'f_avg_Spine1', 'boss:Hips', 'boss:Spine'])
    #
    # retarget(a, b)
    # iterative(a, b)
    # a = {'arm.L':'harm.R', 'hello_l': 'bellow_r', 'a.R': 'aa.R'}
    # print(resolve_left_right(a))
