from __future__ import absolute_import
from __future__ import print_function

# import sys

import os
import fbx
import argparse
from tqdm import tqdm
import numpy as np
import transforms3d
import cv2


# def mat2eulerZXY(mat, unit="deg"):
#     # expects pre_multiply fashion matrix
#     ai, aj, ak = transforms3d.euler.mat2euler(mat, "syxz")
#     # mat = Rz * Rx * Ry
#     if unit == "deg":
#         return np.rad2deg([ak, aj, ai])  # NOTE: why are the orders differ?
#     else:
#         # return np.asarray([ai, aj, ak])  # NOTE: why are the orders differ?
#         return np.asarray([ak, aj, ai])


# def extract_eulerZXY(trans_mats):
#     eulerZXY = np.zeros((trans_mats.shape[0], trans_mats.shape[1], 3))
#     for f in range(trans_mats.shape[0]):
#         for i in range(trans_mats.shape[1]):
#             eulerZXY[f, i, :] = mat2eulerZXY(trans_mats[f, i, :3, :3])
#     return eulerZXY


# def mat2quat(mat):
#     return transforms3d.quaternions.mat2quat(mat)


# def extract_quat(trans_mats):
#     quats = np.zeros((trans_mats.shape[0], trans_mats.shape[1], 4))
#     for f in range(trans_mats.shape[0]):
#         for i in range(trans_mats.shape[1]):
#             quats[f, i, :] = mat2quat(trans_mats[f, i, :3, :3])
#     return quats


# def mat2rvec(mat):
#     # axis, angle = transforms3d.axangles.mat2axangle(mat[:3, :3])
#     # rvec_ = axis * angle
#     # docs is in https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
#     # [rotation vector is a more concise representation of the axis-angle repr](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)
#     rvec, _ = cv2.Rodrigues(mat[:3, :3])
#     # # assert np.all(np.isclose(rvec, rvec_))
#     # sometimes transforms3d throws error: ValueError("no unit eigenvector corresponding to eigenvalue 1")
#     # not sure if cv2 ignores this case or handles it properly. cv2 does not complain about this problem.
#     return rvec.squeeze()


# def extract_rvec(trans_mats):
#     rvecs = np.zeros((trans_mats.shape[0], trans_mats.shape[1], 3))
#     for f in range(trans_mats.shape[0]):
#         for i in range(trans_mats.shape[1]):
#             rvecs[f, i, :] = mat2rvec(trans_mats[f, i, :3, :3])
#     return rvecs


class FbxTools(object):
    def __init__(self, filename=None):
        self.manager = fbx.FbxManager.Create()
        self.scene = fbx.FbxScene.Create(self.manager, "")

        if filename:
            self.import_from_file(filename)

    def close(self):
        self.scene.Destroy()
        self.manager.Destroy()

    def import_from_file(self, filename):
        importer = fbx.FbxImporter.Create(self.manager, "")
        importStatus = importer.Initialize(filename)
        if not importStatus:
            raise Exception(f"Importer failed to initialized.\nError returned: {importer.GetStatus().GetErrorString()}")

        major, minor, revision = importer.GetFileVersion()
        print(f"version numbers of {os.path.basename(filename)}: {major}.{minor}.{revision}")

        importer.Import(self.scene)
        importer.Destroy()

        timeModeStrings = [
            "DefaultMode",
            "Frames120",
            "Frames100",
            "Frames60",
            "Frames50",
            "Frames48",
            "Frames30",
            "Frames30Drop",
            "NTSCDropFrame",
            "NTSCFullFrame",
            "PAL",
            "Frames24",
            "Frames1000",
            "FilmFullFrame",
            "Custom",
            "Frames96",
            "Frames72",
            "Frames59dot94",
        ]

        globalSettings = self.scene.GetGlobalSettings()
        self.timeMode = globalSettings.GetTimeMode()
        self.frameRate = fbx.FbxTime().GetFrameRate(globalSettings.GetTimeMode())
        # assert self.frameRate == 100, "FPS: {}".format(self.frameRate)
        self.timeModeString = timeModeStrings[self.timeMode]

        timeSpan = globalSettings.GetTimelineDefaultTimeSpan()
        self.start, self.stop = timeSpan.GetStart(), timeSpan.GetStop()

        print(f"Global time settings: {self.timeModeString}, {self.frameRate} fps")
        print(f"Timeline default timespan: [{self.start.GetTimeString()}: {self.stop.GetTimeString()}]")
        # self.start.GetTimeString(): FbxString
        tmp = str(self.start.GetTimeString())
        # assert tmp.startswith(("-1", "-0", "0")), "{}".format(tmp)

    def display_animations(self):
        animStackCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        animLayerCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        animCurveNodeCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)

        lines = []
        animStackCount = self.scene.GetSrcObjectCount(animStackCriteria)
        lines.append(f"AnimStack Count: {animStackCount}" + " {")

        for i in range(animStackCount):
            animStack = self.scene.GetSrcObject(animStackCriteria, i)
            lines.append(f"  Stack #{i:02}: {animStack.GetName()}")

            animLayerCount = animStack.GetMemberCount(animLayerCriteria)
            animLayerNames = []
            for j in range(animLayerCount):
                animLayer = animStack.GetMember(animLayerCriteria, j)
                animLayerNames.append(
                    animLayer.GetName() + " " + f"({animLayer.GetMemberCount(animCurveNodeCriteria)} Curves)"
                )
            lines.append(f"  {animLayerCount} layers: {animLayerNames}")

            if i != animStackCount - 1:
                lines.append("")
        lines.append("}")
        return "\n".join(lines)

    def get_first_available_anim_layer(self):
        animStackCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        animLayerCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        animCurveNodeCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)

        animStackCount = self.scene.GetSrcObjectCount(animStackCriteria)
        for i in range(animStackCount):
            animStack = self.scene.GetSrcObject(animStackCriteria, i)
            animLayerCount = animStack.GetMemberCount(animLayerCriteria)
            for j in range(animLayerCount):
                animLayer = animStack.GetMember(animLayerCriteria, j)
                animCurveNodeCount = animLayer.GetMemberCount(animCurveNodeCriteria)

                if animCurveNodeCount > 0:
                    print(
                        f"Selected AnimLayer: [{animStack.GetName()}, {animLayer.GetName()} ({animCurveNodeCount} Curves)]"
                    )
                    return animLayer
        return None

    def get_animlayer_frame_span(self, animLayer):
        animCurveNodeCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        animCurveNodeCount = animLayer.GetMemberCount(animCurveNodeCriteria)

        animLayerStart, animLayerStop = self.start, self.stop
        for i in range(animCurveNodeCount):
            animCurveNode = animLayer.GetMember(animCurveNodeCriteria, i)

            interval = fbx.FbxTimeSpan()
            animCurveNode.GetAnimationInterval(interval)
            start, stop = interval.GetStart(), interval.GetStop()

            animLayerStart = min(animLayerStart, start)
            animLayerStop = max(animLayerStop, stop)

        return (animLayerStart.GetFrameCount(self.timeMode), animLayerStop.GetFrameCount(self.timeMode))

    def get_skeleton_root(self):
        nodeQueue = [self.scene.GetRootNode()]
        while len(nodeQueue) > 0:
            node = nodeQueue.pop(0)

            if isinstance(node.GetNodeAttribute(), fbx.FbxSkeleton):
                return node

            childCount = node.GetChildCount()
            for i in range(childCount):
                child = node.GetChild(i)
                nodeQueue.append(child)

        return None

    def get_skeleton_hierarchy(self):
        skeletonRoot = self.get_skeleton_root()
        hierarchy = []

        def is_zero_rotation(r):
            return np.allclose(r, [0, 0, 0])

        def depth_first_iterate_outliner(node, i, parent, hierarchy):
            localTransform = node.EvaluateLocalTransform()
            T, R = localTransform.GetT(), localTransform.GetR()

            item = {
                "name": node.GetName(),
                "node": node,
                "offset": (T[0], T[1], T[2]),
                "rotation": (R[0], R[1], R[2]),
                "skeleton": isinstance(node.GetNodeAttribute(), fbx.FbxSkeleton),
                "children": [],
                "children_indices": [],
                "parent": parent,
                "parent_index": -1,
                "birth_order": i,
                "is_end_site": False,
                "sibling": 1 if not parent else parent.GetChildCount(),
            }
            hierarchy.append(item)

            childCount = node.GetChildCount()
            for i in range(childCount):
                child = node.GetChild(i)
                item["children"].append(child)
                item["children_indices"].append(-1)
                depth_first_iterate_outliner(child, i, node, hierarchy)

            item["is_end_site"] = (
                len(item["children_indices"]) == 0 and item["birth_order"] == 0 and is_zero_rotation(item["rotation"])
            )

        depth_first_iterate_outliner(skeletonRoot, 0, None, hierarchy)
        nodes = [item["node"] for item in hierarchy]
        for item in hierarchy:
            if item["parent"]:
                index = nodes.index(item["parent"])
                item["parent_index"] = index

            for nth in range(len(item["children"])):
                index = nodes.index(item["children"][nth])
                item["children_indices"][nth] = index

        if not hasattr(self, "hierarchy_checked"):
            nonSkeletonNodes = [item["node"].GetName() for item in hierarchy if not item["skeleton"]]
            if len(nonSkeletonNodes) > 0:
                print(
                    f"Caution! Non-Skeleton nodes exist in the hierarchy ({len(nonSkeletonNodes)} / {len(hierarchy)}, {len(hierarchy) - len(nonSkeletonNodes)})."
                )
                print([item["node"].GetName() for item in hierarchy if not item["skeleton"]])
            self.hierarchy_checked = True

        return hierarchy

    def evaluate_animation_transforms(self):
        animLayer = self.get_first_available_anim_layer()
        start, stop = self.get_animlayer_frame_span(animLayer)

        self.hierarchy = self.get_skeleton_hierarchy()
        # non_endsite_hierarchy = [item for item in self.hierarchy if not item["is_end_site"]]
        non_endsite_hierarchy = self.hierarchy  # NOTE: use this version, the above one is for Tega. Not sure why yet.

        frameCount = stop - start + 1
        jointCount = len(non_endsite_hierarchy)

        self.localTransforms = np.zeros((frameCount, jointCount, 4, 4))
        self.globalTransforms = np.zeros((frameCount, jointCount, 4, 4))

        for f in tqdm(range(start, stop + 1)):
            time = fbx.FbxTime()
            time.SetFrame(f, self.timeMode)

            for i in range(len(non_endsite_hierarchy)):
                node = non_endsite_hierarchy[i]["node"]

                ltrans = node.EvaluateLocalTransform(time)
                gtrans = node.EvaluateGlobalTransform(time)

                lmat = np.transpose(
                    np.array(
                        [
                            [ltrans.Get(0, 0), ltrans.Get(0, 1), ltrans.Get(0, 2), ltrans.Get(0, 3)],
                            [ltrans.Get(1, 0), ltrans.Get(1, 1), ltrans.Get(1, 2), ltrans.Get(1, 3)],
                            [ltrans.Get(2, 0), ltrans.Get(2, 1), ltrans.Get(2, 2), ltrans.Get(2, 3)],
                            [ltrans.Get(3, 0), ltrans.Get(3, 1), ltrans.Get(3, 2), ltrans.Get(3, 3)],
                        ]
                    )
                )  # pre_multiply fashion

                gmat = np.transpose(
                    np.array(
                        [
                            [gtrans.Get(0, 0), gtrans.Get(0, 1), gtrans.Get(0, 2), gtrans.Get(0, 3)],
                            [gtrans.Get(1, 0), gtrans.Get(1, 1), gtrans.Get(1, 2), gtrans.Get(1, 3)],
                            [gtrans.Get(2, 0), gtrans.Get(2, 1), gtrans.Get(2, 2), gtrans.Get(2, 3)],
                            [gtrans.Get(3, 0), gtrans.Get(3, 1), gtrans.Get(3, 2), gtrans.Get(3, 3)],
                        ]
                    )
                )  # pre_multiply fashion

                self.localTransforms[f, i, :, :] = lmat
                self.globalTransforms[f, i, :, :] = gmat

        for item in self.hierarchy:
            del item["node"], item["parent"], item["children"]

        return {
            "hierarchy": self.hierarchy,
            "localTransforms": self.localTransforms,
            "globalTransforms": self.globalTransforms,
        }

    def decouple_animations(self, animations, verbose=True):
        if animations:
            hierarchy = animations["hierarchy"]
            localTransforms = animations["localTransforms"]
            globalTransforms = animations["globalTransforms"]
        elif hasattr(self, "hierarchy") and hasattr(self, "localTransforms") and hasattr(self, "globalTransforms"):
            hierarchy = self.hierarchy
            localTransforms = self.localTransforms
            globalTransforms = self.globalTransforms
        else:
            animations = self.evaluate_animation_transforms()
            hierarchy = animations["hierarchy"]
            localTransforms = animations["localTransforms"]
            globalTransforms = animations["globalTransforms"]
        if verbose:
            print("shape of localTransforms: {}".format(localTransforms.shape))
            print("shape of globalTransforms: {}".format(globalTransforms.shape))
        return hierarchy, localTransforms, globalTransforms

    # def write_to_bvh(self, filename, animations):
    #     hierarchy, localTransforms, globalTransforms = self.decouple_animations(animations)
    #     # names = [item["name"].strip() for item in hierarchy]

    #     frames = np.zeros((localTransforms.shape[0], 3 + localTransforms.shape[1] * 3))
    #     frames[:, :3] = globalTransforms[:, 0, :3, -1]

    #     for f in range(localTransforms.shape[0]):
    #         for i in range(localTransforms.shape[1]):
    #             mat = localTransforms[f, i, :, :]
    #             frames[f, 3 + i * 3 : 3 + (i + 1) * 3] = mat2eulerZXY(mat[:3, :3])

    #     result = Bvh()
    #     result.update_hierarchy_section(hierarchy)

    #     result.frame_time = 1.0 / self.frameRate
    #     result.nframes = localTransforms.shape[0]
    #     # result.frames = frames.astype(str)
    #     result.frames = np.require(frames, dtype=str)
    #     with open(filename, "w") as f:
    #         f.write(str(result))

    def extract_data(self, animations):
        hierarchy, localTransforms, globalTransforms = self.decouple_animations(animations)
        names = [item["name"].strip() for item in hierarchy]

        # global_translation = globalTransforms[:, 0, :3, -1]
        # xyz = np.zeros((globalTransforms.shape[0], len(hierarchy), 3))
        # for f in range(globalTransforms.shape[0]):
        #     for i in range(globalTransforms.shape[1]):
        #         xyz[f, i, :] = globalTransforms[f, i, :3, -1]
        xyz = globalTransforms[:, :, :3, -1]
        global_translation = globalTransforms[:, 0, :3, -1]

        trans_mats = localTransforms.copy()
        trans_mats[:, 0, :, :] = globalTransforms[:, 0, :, :]

        return names, xyz, trans_mats, global_translation


# def parse_cmd_line_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", "-i", type=str, required=False, help="the path of the input .fbx file.")
#     parser.add_argument("--overwrite", "-o", action="store_true", help="")
#     parser.add_argument(
#         "--bvh", "-b", type=str, default=None, help="the output path of the .bvh format animation data."
#     )
#     parser.add_argument("--filename", "-f", type=str, default=None, help="the output h5 file path")
#     parser.add_argument("--trans_mats", "-t", action="store_true", help="store 4x4 transformation matrix")
#     parser.add_argument("--rvecs", "-r", action="store_true", help="store rotation vectors (axis times angle)")
#     parser.add_argument("--quats", "-q", action="store_true", help="store quaternions")
#     parser.add_argument("--eulerZXY", "-e", action="store_true", help="store Euler angles in z,x,y order")
#     parser.add_argument("--start", "-s", type=int, default=0, help="from which the valid data start")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_cmd_line_args()
