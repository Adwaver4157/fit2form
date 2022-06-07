import os
import sys
from argparse import ArgumentParser
sys.path.append('/user/fit2form')
from environment.GraspSimulationEnv import *
from utils import dicts_get, load_config, load_evaluate_config
from learning import ObjectDataset
import numpy as np
from tqdm import tqdm
from learning.finger_generator import get_finger_generator, ImprintFingerGenerator
from shutil import copyfile
from os.path import splitext
from json import dump
import pickle

from torch.utils.data import DataLoader
from common_utils import glob_category
from os.path import exists, splitext
from pathlib import Path

def tqdm_get(task_handles, desc=None, max_results=None):
    results = []
    if max_results is None:
        max_results = len(task_handles)
    with tqdm(total=max_results, desc=desc, dynamic_ncols=True) as pbar:
        pbar.update(0)
        for task in task_handles:
            results.append(task)
            pbar.update(1)
    return results

class test_ObjectDataset(ObjectDataset):
    def __init__(self,
                 directory_path: str,
                 batch_size: int,
                 category_file: str = None,
                 check_validity=True):
        if category_file is not None and exists(category_file):
            print('[ObjectDataset] loading grasp objects from ',
                  directory_path, category_file)
            self.object_paths = glob_category(
                directory_path, category_file, "**/*graspobject.urdf")
        else:
            print('[ObjectDataset] globbing grasp objects ...')
            self.object_paths = [str(p)
                                 for p in Path(directory_path).rglob("*graspobject.urdf")]
        self.directory_path = directory_path
        total_count = len(self.object_paths)
        print(f'[ObjectDataset] found {total_count} objects')
        if check_validity:
            self.object_paths = [
                path for path in tqdm_get(
                    task_handles=[
                        self.validity_filter(object_path)
                        for object_path in self.object_paths
                    ],
                    desc="checking grasp objects ..."
                )
                if path is not None
            ]
            valid_count = len(self.object_paths)
            print('[ObjectDataset] found {} bad objects'.format(
                total_count - valid_count))
        self.batch_size = batch_size
        self.get_urdf_path_only = False
        self.loader = DataLoader(
            self,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=lambda batch: batch)
        self.loader_iter = iter(self.loader)
        self.iter_position = 0

def acc_results(results):
    metrics = {
        'base_connected': np.empty(0),
        'created_grippers_failed': np.empty(0),
        'single_connected_component': np.empty(0),
        'grasp_object_path': []
    }
    for key in results[0]['score'][0].keys():
        metrics[key] = np.empty(0)
    for r in results:
        if r["score"] is None:
            print("Score is none. Ignoring ...")
            continue
        metrics['base_connected'] = np.concatenate(
            (metrics['base_connected'],
                [r['base_connected']]))
        metrics['created_grippers_failed'] = np.concatenate(
            (metrics['created_grippers_failed'],
                [r['created_grippers_failed']]))
        metrics['single_connected_component'] = np.concatenate(
            (metrics["single_connected_component"],
                [r["single_connected_component"]]))
        metrics['grasp_object_path'].append(r['grasp_object_path'])
        for score in r["score"]:
            for key, r_val in score.items():
                metrics[key] = np.concatenate((metrics[key], [r_val]))
    for key, val in metrics.items():
        if key is not 'grasp_object_path':
            metrics[key] = metrics[key].astype(float)
    return metrics


def print_results(metrics, name=""):
    print(f"Results summary {name}:")
    for key, val in metrics.items():
        if key != 'grasp_object_path':
            print(f"{key}: {np.mean(np.array(val, dtype=float)):.4f}")

def test_compute_finger_grasp_score(env,
                                    left_finger_tsdf,
                                    right_finger_tsdf,
                                    grasp_object_urdf_paths,
                                    urdf_output_path_prefix,
                                    visualize=False):
        scores = []
        if not check_base_connection(right_finger_tsdf)\
                or not check_base_connection(left_finger_tsdf):
            for grasp_object_urdf_path in grasp_object_urdf_paths:
                scores.append(env.failure_score())
            return {
                'score': scores,
                'base_connected': False,
                'created_grippers_failed': None,
                'single_connected_component': None,
                'grasp_object_path': grasp_object_urdf_path
            }
        # 1. create gripper meshes
        retval = create_gripper(
            left_finger=left_finger_tsdf.squeeze(),
            right_finger=right_finger_tsdf.squeeze(),
            urdf_output_path_prefix=urdf_output_path_prefix,
            voxel_size=env.tsdf_voxel_size)
        if retval is None:
            for grasp_object_urdf_path in grasp_object_urdf_paths:
                scores.append(env.failure_score())
            return {
                'score': scores,
                'base_connected': True,
                'created_grippers_failed': True,
                'single_connected_component': None,
                'grasp_object_path': grasp_object_urdf_path
            }
        gripper_urdf_path, single_connected_component = retval

        # 2. simulate the grippers
        for grasp_object_urdf_path in grasp_object_urdf_paths:
            scores.append(env.simulate_grasp(
                    grasp_object_urdf_path=grasp_object_urdf_path,
                    gripper_urdf_path=gripper_urdf_path,
                    left_finger_tsdf=left_finger_tsdf,
                    right_finger_tsdf=right_finger_tsdf,
                    visualize=visualize))
        return {
            'score': scores,
            'base_connected': True,
            'created_grippers_failed': False,
            'single_connected_component': single_connected_component,
            'grasp_object_path': grasp_object_urdf_path
        }

if __name__ == '__main__':
    parser = ArgumentParser('Generator fingers evaluater')
    parser.add_argument("--evaluate_config",
                        help="path to evaluate config file",
                        required=True,
                        type=str)
    parser.add_argument('--objects',
                        help='path to shapenet root',
                        type=str,
                        required=True)
    parser.add_argument('--config',
                        help='path to JSON config file',
                        type=str,
                        default='configs/default.json')
    parser.add_argument('--name',
                        help='path to directory to save results',
                        type=str, required=True)
    parser.add_argument('--num_processes',
                        help='number of environment processes',
                        type=int,
                        default=32)
    parser.add_argument('--gui', action='store_true',
                        default=False, help='Run headless or render')
    parser.add_argument('--objects_bs',
                        help='objects batch size',
                        type=int,
                        default=128)
    args = parser.parse_args()

    # load environment
    config = load_config(args.config)
    env = GraspSimulationEnv(
        config=config,
        gui=args.gui,
        visualize=False)

    # object dataset
    obj_dataset = test_ObjectDataset(
        directory_path=args.objects,
        batch_size=args.objects_bs,
    )
    grippers_directory = args.name + '/'
    # assert not os.path.exists(grippers_directory)
    os.makedirs(grippers_directory, exist_ok=True)

    # evaluations
    evaluate_config = load_evaluate_config(args.evaluate_config)

    # dump config and args
    print("logging to ", grippers_directory)
    dump(config, open(grippers_directory + 'config.json', 'w'), indent=4)
    dump(evaluate_config, open(grippers_directory + 'evaluate_config.json', 'w'), indent=4)
    pickle.dump(args,open(grippers_directory + 'args.pkl','wb'))

    finger_generators = []
    for finger_generator_config in evaluate_config:
        if not finger_generator_config["evaluate"]:  # TODO: Do it more neatly
            continue
        results = []
        finger_generator = get_finger_generator(finger_generator_config)
        print(finger_generator)
        obj_loader = iter(obj_dataset.loader)
        for i, grasp_objects in enumerate(tqdm(obj_loader, smoothing=0.01, dynamic_ncols=True, desc=finger_generator_config["desc"])):
            print(i)
            retval = finger_generator.create_fingers(
                grasp_objects)
            gripper_output_paths = ['{}/{:06}'.format(
                grippers_directory, i + len(results))
                for i in range(len(grasp_objects))]
            if type(finger_generator) == ImprintFingerGenerator:
                left_fingers, right_fingers, gripper_urdf_paths = retval
                for gripper_urdf_path, target_gripper_urdf_path in \
                        zip(gripper_urdf_paths, gripper_output_paths):
                    # copy collision meshes over
                    prefix = splitext(gripper_urdf_path)[0]
                    copyfile(prefix + '_right_collision.obj',
                             target_gripper_urdf_path + '_right_collision.obj')
                    copyfile(prefix + '_left_collision.obj',
                             target_gripper_urdf_path + '_left_collision.obj')
            else:
                left_fingers, right_fingers = retval
            grasp_results = []
            for left, right, value, path, flag in zip(
                    left_fingers,
                    right_fingers,
                    dicts_get(grasp_objects, 'urdf_path'),
                    gripper_output_paths,
                    [False] * len(gripper_output_paths)):
                result = test_compute_finger_grasp_score(env, left, right, dicts_get(grasp_objects, 'urdf_path'), path, flag)
                grasp_results.append(result)
            results.extend(grasp_results)
            metrics = acc_results(results)
            scores = metrics.reshape((len(grasp_objects), -1))
            print(scores)
            print_results(metrics, name=finger_generator_config["desc"])
            save_path = os.path.join(grippers_directory, "val_results.npz")
            np.savez(save_path, **metrics)
            print(f"saved results at {save_path}")
