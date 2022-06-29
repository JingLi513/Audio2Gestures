"""
    smplx2fbx.py
        > Convert SmplX model to FBX
"""

import sys
import fbx
import pickle
import argparse
import json
import h5py
import numpy as np

# import torch
import transforms3d
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.axangles import axangle2mat


# yapf: enable
pass
# yapf: disable
smplx_joint2num = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
    "L_Ankle": 7, "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11,
    "Neck": 12, "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17,
    "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "Jaw": 22, "L_Eye": 23, "R_Eye": 24,
    "L_Index1": 25, "L_Index2": 26, "L_Index3": 27,
    "L_Middle1": 28, "L_Middle2": 29, "L_Middle3": 30,
    "L_Pinky1": 31, "L_Pinky2": 32, "L_Pinky3": 33,
    "L_Ring1": 34, "L_Ring2": 35, "L_Ring3": 36,
    "L_Thumb1": 37, "L_Thumb2": 38, "L_Thumb3": 39,
    "R_Index1": 40, "R_Index2": 41, "R_Index3": 42,
    "R_Middle1": 43, "R_Middle2": 44, "R_Middle3": 45,
    "R_Pinky1": 46, "R_Pinky2": 47, "R_Pinky3": 48,
    "R_Ring1": 49, "R_Ring2": 50, "R_Ring3": 51,
    "R_Thumb1": 52, "R_Thumb2": 53, "R_Thumb3": 54,
}  # this is the order when you create FBX file
# yapf: enable
pass
# yapf: disable
# smplx_joints = [
#     "Pelvis", "L_Hip", "R_Hip", "Spine1",
#     "L_Knee", "R_Knee", "Spine2",
#     "L_Ankle", "R_Ankle",
#     "Spine3",
#     "L_Foot", "R_Foot",
#     "Neck", "L_Collar", "R_Collar", "Head",
#     "L_Shoulder", "R_Shoulder",
#     "L_Elbow", "R_Elbow",
#     "L_Wrist", "R_Wrist",
#     "Jaw", "L_Eye", "R_Eye",
#     "L_Index1", "L_Index2", "L_Index3",
#     "L_Middle1", "L_Middle2", "L_Middle3",
#     "L_Pinky1", "L_Pinky2", "L_Pinky3",
#     "L_Ring1", "L_Ring2", "L_Ring3",
#     "L_Thumb1", "L_Thumb2", "L_Thumb3",
#     "R_Index1", "R_Index2", "R_Index3",
#     "R_Middle1", "R_Middle2", "R_Middle3",
#     "R_Pinky1", "R_Pinky2", "R_Pinky3",
#     "R_Ring1", "R_Ring2", "R_Ring3",
#     "R_Thumb1", "R_Thumb2", "R_Thumb3",
# ]  # this is the order when you create FBX file
# yapf: enable
pass
# yapf: disable
smplx_kintree_table = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5,
    6,
    7, 8,
    9, 9, 9, 12,
    13, 14,
    16, 17,
    18, 19,
    15, 15, 15,
    20, 25, 26,
    20, 28, 29,
    20, 31, 32,
    20, 34, 35,
    20, 37, 38,
    21, 40, 41,
    21, 43, 44,
    21, 46, 47,
    21, 49, 50,
    21, 52, 53,
]  # this is for smplx_joint2num
# yapf: enable
pass
# yapf: disable
# smpl_joint_names = [
#     'Pelvis',
#     'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot',
#     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot',
#     'Spine1', 'Spine2', 'Spine3',
#     'Neck', 'Head',
#     'L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
#     'R_Collar', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
# ]  # this is only used for exporting fbx data into HDF5 since the joints are traversed in this order
# yapf: enable
pass
# yapf: disable
smpl_joint2num = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6, "L_Ankle": 7, "R_Ankle": 8, "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11,
    "Neck": 12, "L_Collar": 13, "R_Collar": 14,
    "Head": 15, "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "L_Hand": 22, "R_Hand": 23,
}
# yapf: enable

pass

# yapf: disable
smpl_kintree_table = [
    -1, 0, 0, 0,
    1, 2, 3, 4, 5, 6,
    7, 8,
    9, 9, 9,
    12, 13, 14, 16, 17,
    18, 19,
    20, 21
]
# yapf: enable



def addSmplXMesh(fbxScene, v_posed, faces):
    # Obtain a reference to the scene's root node.
    rootNode = fbxScene.GetRootNode()

    # Create a new node in the scene.
    geometryNode = fbx.FbxNode.Create(fbxScene, "Geometry")
    rootNode.AddChild(geometryNode)

    # Create a new mesh node attribute in the scene, and
    # set it as the new node's attribute
    mesh = fbx.FbxMesh.Create(fbxScene, "body")
    geometryNode.SetNodeAttribute(mesh)

    # Define the new mesh's control points.
    # v_posed, faces = smplx['v_posed'], smplx['faces']
    v_posed = np.array(v_posed)
    faces = np.array(faces)

    minValue = np.min(v_posed)
    maxValue = np.max(v_posed)
    # print(f"min = {minValue}, max = {maxValue}")
    print("min = {}, max = {}".format(minValue, maxValue))

    # m = axangle2mat((1, 0, 0), np.radians(180))

    mesh.InitControlPoints(v_posed.shape[0])
    for i in range(v_posed.shape[0]):
        v = v_posed[i, :]
        # v = np.matmul(m, v)
        vertex = fbx.FbxVector4(v[0], v[1], v[2])
        mesh.SetControlPointAt(vertex, i)

    for i in range(faces.shape[0]):
        mesh.BeginPolygon(i)
        mesh.AddPolygon(faces[i, 0])
        mesh.AddPolygon(faces[i, 1])
        mesh.AddPolygon(faces[i, 2])
        mesh.EndPolygon()

    return geometryNode


def addSmplXSkeleton(fbxScene, trans, joint2num, kintree_table):
    num2joint = ["" for key in joint2num]
    for key, value in joint2num.items():
        num2joint[value] = key

    # trans = np.array(trans)

    # Obtain a reference to the scene's root node.
    rootNode = fbxScene.GetRootNode()

    # Create a new node in the scene.
    referenceNode = fbx.FbxNode.Create(fbxScene, "Reference")
    rootNode.AddChild(referenceNode)

    # Create skeletons
    skeletonNodes = []
    for nth in range(len(kintree_table)):
        skeleton = fbx.FbxSkeleton.Create(fbxManager, "")
        skeleton.SetSkeletonType(fbx.FbxSkeleton.eRoot if nth == -1 else fbx.FbxSkeleton.eLimbNode)

        node = fbx.FbxNode.Create(fbxScene, num2joint[nth])
        node.SetNodeAttribute(skeleton)

        # trans = transforms_mat[nth][0:3, 3]
        # rotation = mat2euler(transforms_mat[nth][0:3, 0:3])
        # rotation = np.rad2deg(rotation)

        node.LclTranslation.Set(fbx.FbxDouble3(trans[nth,0], trans[nth, 1], trans[nth,2]))
        # node.LclRotation.Set(fbx.FbxDouble3(rotation[0], rotation[1], rotation[2]))
        # node.LclScaling.Set(
        #     fbx.FbxDouble3(transforms_mat[nth][0, 0], transforms_mat[nth][1, 1], transforms_mat[nth][2, 2])
        # )

        skeletonNodes.append(node)

        if kintree_table[nth] != -1:
            skeletonNodes[kintree_table[nth]].AddChild(node)

    referenceNode.AddChild(skeletonNodes[0])
    return referenceNode, skeletonNodes


def addSkiningWeight(fbxScene, smplx, geometryNode, skeletonNodes):
    lbs_weights = smplx["lbs_weights"]
    # print(lbs_weights.shape)
    # print(lbs_weights)

    clusters = []
    for i in range(lbs_weights.shape[1]):
        cluster = fbx.FbxCluster.Create(fbxScene, "")
        cluster.SetLink(skeletonNodes[i])
        cluster.SetLinkMode(fbx.FbxCluster.eTotalOne)

        for j in range(lbs_weights.shape[0]):
            weight = lbs_weights[j, i]
            if weight > 0:
                cluster.AddControlPointIndex(j, weight)

        clusters.append(cluster)
        # print(cluster.GetControlPointIndicesCount())

    # Now we have the Geometry and the skeleton correctly positioned,
    # set the transform and TransformLink matrix accordingly.
    matrix = fbxScene.GetAnimationEvaluator().GetNodeGlobalTransform(geometryNode)
    for cluster in clusters:
        cluster.SetTransformMatrix(matrix)

    for i in range(len(skeletonNodes)):
        matrix = fbxScene.GetAnimationEvaluator().GetNodeGlobalTransform(skeletonNodes[i])
        clusters[i].SetTransformLinkMatrix(matrix)

    # Add the clusters to the patch by creating a skin and adding those clusters to that skin.
    skin = fbx.FbxSkin.Create(fbxScene, "")
    for cluster in clusters:
        skin.AddCluster(cluster)
    geometryNode.GetNodeAttribute().AddDeformer(skin)


def storeBindPose(fbxScene, geometryNode):
    # In the bind pose, we must store all the link's global matrix at the
    # time of the bind.
    # Plus, we must store all the parent(s) global matrix of a link, even
    # if they are not themselves deforming any model.

    clusteredNodes = []
    if geometryNode and geometryNode.GetNodeAttribute():
        skinCount = 0
        clusterCount = 0
        attributeType = geometryNode.GetNodeAttribute().GetAttributeType()
        if attributeType in (fbx.FbxNodeAttribute.eMesh, fbx.FbxNodeAttribute.eNurbs, fbx.FbxNodeAttribute.ePatch):
            skinCount = geometryNode.GetNodeAttribute().GetDeformerCount(fbx.FbxDeformer.eSkin)
            for i in range(skinCount):
                skin = geometryNode.GetNodeAttribute().GetDeformer(i, fbx.FbxDeformer.eSkin)
                clusterCount += skin.GetClusterCount()

        if clusterCount:
            for i in range(skinCount):
                skin = geometryNode.GetNodeAttribute().GetDeformer(i, fbx.FbxDeformer.eSkin)
                clusterCount = skin.GetClusterCount()
                for j in range(clusterCount):
                    link = skin.GetCluster(j).GetLink()
                    addNodeRecursively(clusteredNodes, link)

            # Add the geometry to the pose
            clusteredNodes += [geometryNode]

    # Now create a bind pose with the link list
    if len(clusteredNodes):
        # A pose must be named. Arbitrarily use the name of the geometry node.
        pose = fbx.FbxPose.Create(fbxScene, geometryNode.GetName())
        pose.SetIsBindPose(True)

        for node in clusteredNodes:
            bindMatrix = fbxScene.GetAnimationEvaluator().GetNodeGlobalTransform(node)
            pose.Add(node, fbx.FbxMatrix(bindMatrix))

        fbxScene.AddPose(pose)


def addNodeRecursively(nodeArray, node):
    """
        Add the specified node to the node array. Also, add recursively
        all the parent node of the specified node to the array.
    """
    if node:
        addNodeRecursively(nodeArray, node.GetParent())
        found = False
        if node in nodeArray:
            if node.GetName() == node.GetName():
                found = True
        if not found:
            nodeArray += [node]


def batch_rodrigues_np(rot_vecs, epsilon=1e-8, dtype=np.float64):
    """ Calculates the rotation matrices for a batch of rotation vectors (axis-angle)
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]

    angle = np.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle

    cos = np.expand_dims(np.cos(angle), axis=1)
    sin = np.expand_dims(np.sin(angle), axis=1)

    # Bx1 arrays
    rx, ry, rz = np.split(rot_dir, 3, axis=1)  # NOTE: the default behavior of torch.split and np.split is different
    K = np.zeros((batch_size, 3, 3), dtype=dtype)

    zeros = np.zeros((batch_size, 1), dtype=dtype)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape((batch_size, 3, 3))

    ident = np.expand_dims(np.eye(3, dtype=dtype), axis=0)
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)
    return rot_mat


def animateGlobalTransformsFromTransMat(animLayer, referenceNode, global_trans_mat, global_translation, frameDuration):
    # global_orient = []
    # camera_transl = []

    # for nth in range(len(frames)):
    #     global_orient.append(np.array(frames[nth][0]["global_orient"]))
    #     camera_transl.append(np.array(frames[nth][0]["camera_transl"]))

    # global_orient = np.stack(global_orient, axis=0)
    # global_orient = batch_rodrigues_np(global_orient)
    # for nth in range(global_orient.shape[0]):
    # global_orient[nth, :, :] = axangle2mat((1, 0, 0), np.radians(180)) * global_orient[nth, :, :]
    global_orient = global_trans_mat[:, :3, :3]
    if global_translation is not None:
        camera_transl = global_translation
    else:
        if global_trans_mat.shape[1:] == (4, 4):
            camera_transl = global_trans_mat[:, :3, 3]
        else:
            print(">>>> NOTE: no global translation")
            camera_transl = np.zeros((len(global_trans_mat), 3))

    rotations = []
    for i in range(global_orient.shape[0]):
        rotation = mat2euler(global_orient[i, :, :])
        rotation = np.rad2deg(rotation)
        rotations.append(rotation)

    # animateSingleChannel(animLayer, referenceNode.LclRotation, "X", rotations, frameDuration)
    # animateSingleChannel(animLayer, referenceNode.LclRotation, "Y", rotations, frameDuration)
    # animateSingleChannel(animLayer, referenceNode.LclRotation, "Z", rotations, frameDuration)

    animateSingleChannel(animLayer, referenceNode.LclTranslation, "X", camera_transl, frameDuration)
    animateSingleChannel(animLayer, referenceNode.LclTranslation, "Y", camera_transl, frameDuration)
    animateSingleChannel(animLayer, referenceNode.LclTranslation, "Z", camera_transl, frameDuration)


def animateSingleChannel(animLayer, component, name, values, frameDuration):
    ncomp = 0

    if name == "X":
        ncomp = 0
    elif name == "Y":
        ncomp = 1
    elif name == "Z":
        ncomp = 2

    time = fbx.FbxTime()
    curve = component.GetCurve(animLayer, name, True)
    curve.KeyModifyBegin()
    for nth in range(len(values)):
        time.SetSecondDouble(nth * frameDuration)
        keyIndex = curve.KeyAdd(time)[0]
        curve.KeySetValue(keyIndex, values[nth][ncomp])
        curve.KeySetInterpolation(
            keyIndex, fbx.FbxAnimCurveDef.eInterpolationConstant
        )  # NOTE: using eInterpolationCubic to do interpolation causes error.
    curve.KeyModifyEnd()


def animateRotationKeyFrames(animLayer, node, transforms_mat, frameDuration):
    rotations = []
    for nth in range(len(transforms_mat)):
        rotations.append(np.rad2deg(mat2euler(transforms_mat[nth][0:3, 0:3])))

    animateSingleChannel(animLayer, node.LclRotation, "X", rotations, frameDuration)
    animateSingleChannel(animLayer, node.LclRotation, "Y", rotations, frameDuration)
    animateSingleChannel(animLayer, node.LclRotation, "Z", rotations, frameDuration)


def animateTranslationKeyFrames(animLayer, node, transforms_mat, frameDuration):
    translations = []
    for nth in range(len(transforms_mat)):
        translations.append(transforms_mat[nth][0:3, 3])

    animateSingleChannel(animLayer, node.LclTranslation, "X", translations, frameDuration)
    animateSingleChannel(animLayer, node.LclTranslation, "Y", translations, frameDuration)
    animateSingleChannel(animLayer, node.LclTranslation, "Z", translations, frameDuration)


def animateScalingKeyFrames(animLayer, node, transforms_mat, frameDuration):
    scalings = []
    for nth in range(len(transforms_mat)):
        scalings.append(np.array((transforms_mat[nth][0, 0], transforms_mat[nth][1, 1], transforms_mat[nth][2, 2])))

    animateSingleChannel(animLayer, node.LclTranslation, "X", scalings, frameDuration)
    animateSingleChannel(animLayer, node.LclTranslation, "Y", scalings, frameDuration)
    animateSingleChannel(animLayer, node.LclTranslation, "Z", scalings, frameDuration)



def animateSkeleton(
    fbxScene, skeletonNodes, frames, frameRate, name="Take1"
):
    frameDuration = 1.0 / frameRate

    if name != "Take1":
        subs = name.split("/")
        name = subs[-1][:-5]

    animStack = fbx.FbxAnimStack.Create(fbxScene, name)
    animLayer = fbx.FbxAnimLayer.Create(fbxScene, "Base Layer")
    animStack.AddMember(animLayer)
    # animateGlobalTransformsFromTransMat(
    #     animLayer=animLayer,
    #     referenceNode=referenceNode,
    #     global_trans_mat=global_trans_mat,
    #     global_translation=global_translation,
    #     frameDuration=frameDuration,
    # )

    for nId in range(len(skeletonNodes)):
        animateRotationKeyFrames(
            animLayer=animLayer, node=skeletonNodes[nId], transforms_mat=frames[:, nId], frameDuration=frameDuration
        )



def saveScene(filename, fbxManager, fbxScene):
    exporter = fbx.FbxExporter.Create(fbxManager, "")
    isInitialized = exporter.Initialize(filename)

    if isInitialized is False:
        raise Exception(
            "Exporter failed to initialized. Error returned: {}".format(exporter.GetStatus().GetErrorString())
        )

    exporter.Export(fbxScene)
    exporter.Destroy()


def parseCmdLines():
    parser = argparse.ArgumentParser(description="SMPLX to FBX Converter")

    parser.add_argument("--smplx", required=True, type=str, help="path to the SMPLX exported model file")
    parser.add_argument("--fbx", required=True, type=str, help="path to the saved FBX file")
    parser.add_argument(
        "--key", required=True, type=str, help="when using HDF data, which dataset to use for the following processing"
    )
    parser.add_argument("--fps", default=None, type=float, help="frame rate of the input MoCap data (CHECK again)")
    parser.add_argument("--synthesized", required=True, type=str, help="path to the synthesized json file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseCmdLines()

    smplx_fn = args.smplx
    fbx_fn = args.fbx

    print("smplx_fn: {}".format(smplx_fn))
    print("motion HDF fn: {}".format(args.synthesized))
    print("fbx_fn: {}".format(fbx_fn))

    with h5py.File(smplx_fn, "r") as f:
        smplx = dict()
        smplx["lbs_weights"] = f["lbs_weights"][()]
        smplx["faces"] = f["faces"][()]
        smplx["v_posed"] = f["v_posed"][()]
        smplx["trans"] = f["trans"][()]
    if len(smplx["lbs_weights"]) == 6890:
        model_type = "SMPL"
        joint2num = smpl_joint2num
        kintree_table = smpl_kintree_table
    elif len(smplx["lbs_weights"]) == 10475:
        model_type = "SMPL-X"
        joint2num = smplx_joint2num
        kintree_table = smplx_kintree_table
    else:
        print("Error")
        sys.exit(1)
    print("model_type: {}".format(model_type))

    assert args.key is not None
    with h5py.File(args.synthesized, "r") as f:
        print("keys: {}".format(list(f.keys())))
        frames = f[args.key][()]

        if "framerate" in f:
            fps = f["framerate"][()]
            if args.fps is not None:
                assert fps == args.fps
        else:
            fps = args.fps

    fbxManager = fbx.FbxManager.Create()
    fbxScene = fbx.FbxScene.Create(fbxManager, "")
    timeMode = fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
    fbxScene.GetGlobalSettings().SetTimeMode(timeMode)

    geometryNode = addSmplXMesh(fbxScene, smplx["v_posed"], smplx["faces"])
    referenceNode, skeletonNodes = addSmplXSkeleton(
        fbxScene=fbxScene, trans=smplx["trans"], joint2num=joint2num, kintree_table=kintree_table
    )

    addSkiningWeight(fbxScene, smplx, geometryNode, skeletonNodes)
    storeBindPose(fbxScene, geometryNode)
    animateSkeleton(
        fbxScene=fbxScene,
        skeletonNodes=skeletonNodes,
        frames=frames,
        frameRate=fps,
    )

    # Save the scene.
    if fps != 25:
        fbx_fn = fbx_fn.replace(".fbx", "_{}fps.fbx".format(int(fps)))
    saveScene(fbx_fn, fbxManager, fbxScene)

    # CLEANUP
    fbxManager.Destroy()
    del fbxManager, fbxScene
