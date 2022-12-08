from concurrent.futures import ThreadPoolExecutor
import json
import os
from dataset_utils.abstract_viz import AbstractViz
from pointcloud_registration.framedata import FrameData, load_frame_data_from_dataset
from pointcloud_registration.pointcloud_registration import PairwiseRegistrationLog
from typing import Dict, List, Optional
from fire import Fire
import open3d.visualization.gui as gui
import numpy as np
from tqdm import tqdm


def fts(f: float, precision: int):
    """
    Formats a float to a string with a fixed number after the decimal point.
    adds a space before the number if it is positive.
    If f < 10, adds one more space before the number.
    """
    if f < 0:
        return '{:.{}f}'.format(f, precision)
    elif f < 10: # positive
        return '  {:.{}f}'.format(f, precision)
    else:
        return ' {:.{}f}'.format(f, precision)


class RegistrationViz(AbstractViz):
    def __init__(self, registrations: List[PairwiseRegistrationLog], dataset_dir: Optional[str] = None):
        super().__init__()
        self.pairwise_registrations: List[PairwiseRegistrationLog] = registrations
        self.add_settings_panel_section('pairwise_registration', 'Pairwise Registration')
        self.slider = self.add_slider_selection('pairwise_registration', (0, len(registrations) - 1), self.change_pairwise_registration_slider)

        self.label_source_id = gui.Label(f'Source ID: {self.pairwise_registrations[0].source.frames[0].frame_data["image_id"]}')
        self.label_target_id = gui.Label(f'Target ID: {self.pairwise_registrations[0].target.frames[0].frame_data["image_id"]}')
        self.label_type = gui.Label(f'Type: {self.pairwise_registrations[0].edge_type}')
        self.matrix_label_rows = None

        self.add_settings_panel_child("pairwise_registration", self.label_source_id)
        self.add_settings_panel_child("pairwise_registration", self.label_target_id)
        self.add_settings_panel_child("pairwise_registration", self.label_type)
        self.update_transform_matrix_label(self.pairwise_registrations[0].transform_matrix)

        self.iteration_slider = self.add_slider_selection('pairwise_registration', (0, 1), self.change_iteration_slider)

        self.change_pairwise_registration_slider(self.slider, 0)
        self.add_show_checkbox_for_geometry('pairwise_registration_pcd_source', 'pairwise_registration', "Show Source", True)
        self.add_show_checkbox_for_geometry('pairwise_registration_pcd_target', 'pairwise_registration', "Show Target", True)


    def update_transform_matrix_label(self, matrix: np.ndarray):
        if self.matrix_label_rows is None:
            self.matrix_label_rows = []
            for i in range(4):
                row = gui.Label(f'{fts(matrix[i, 0], 3)}\t{fts(matrix[i, 1], 3)}\t{fts(matrix[i, 2], 3)}\t{fts(matrix[i, 3], 3)}')
                self.add_settings_panel_child("pairwise_registration", row)
                self.matrix_label_rows.append(row)

        else:
            for i in range(4):
                self.matrix_label_rows[i].text = f'{fts(matrix[i, 0], 3)}\t{fts(matrix[i, 1], 3)}\t{fts(matrix[i, 2], 3)}\t{fts(matrix[i, 3], 3)}'


    def change_pairwise_registration_slider(self, slider, value: int):
        registration = self.pairwise_registrations[value]

        self.remove_geometry('pairwise_registration_pcd_source')
        self.remove_geometry('pairwise_registration_pcd_target')

        self.add_geometry('pairwise_registration_pcd_source', registration.source.pointcloud.as_open3d())
        # self._scene.scene.set_geometry_transform('pairwise_registration_pcd_source', registration.transform_matrix)
        self.add_geometry('pairwise_registration_pcd_target', registration.target.pointcloud.as_open3d())

        self.label_source_id.text = f'Source ID: {registration.source.frames[0].frame_data["image_id"]}'
        self.label_target_id.text = f'Target ID: {registration.target.frames[0].frame_data["image_id"]}'
        self.label_type.text = f'Type: {registration.edge_type}'
        # self.update_transform_matrix_label(registration.transform_matrix)

        self.iteration_slider.set_limits(0, len(registration.iteration_data))
        self.iteration_slider.int_value = len(registration.iteration_data)

        self.change_iteration_slider(self.iteration_slider, len(registration.iteration_data))


    def change_iteration_slider(self, slider, value: int):
        registration = self.pairwise_registrations[self.slider.int_value]
        iteration_data = ([{
            "transformation": np.eye(4),
            "inlier_rmse": -1,
            "scale_index": -1,
            "fitness": -1,
            "scale_iteration_index": -1,
            "iteration_index": -1,
        }] + registration.iteration_data)[value]

        # print(iteration_data)

        self._scene.scene.set_geometry_transform('pairwise_registration_pcd_source', iteration_data["transformation"])
        self.update_transform_matrix_label(iteration_data["transformation"])

def load_frames_dict(dataset_dir: str, json_file_name: str) -> Dict[int, FrameData]:
    frames = load_frame_data_from_dataset(dataset_dir, json_file_name)
    return {frame.frame_data["image_id"]: frame for frame in frames}

def main(dataset_dir: str):
    pairwise_registration_file = os.path.join(dataset_dir, 'pairwise_registrations.json')
    if not os.path.exists(pairwise_registration_file):
        print('Pairwise registration file does not exist: {}'.format(pairwise_registration_file))
        return

    with open(pairwise_registration_file, 'r') as f:
        data = json.load(f)

    json_file_name = data["transforms_json_file"]
    frames_dict = load_frames_dict(dataset_dir, json_file_name)

    with ThreadPoolExecutor() as executor:
        pairwise_registrations = list(tqdm(executor.map(lambda pfd: PairwiseRegistrationLog.from_dict(pfd, frames_dict), data["pairwise_registrations"])))

    gui.Application.instance.initialize()
    viz = RegistrationViz(pairwise_registrations)
    gui.Application.instance.run()

if __name__ == '__main__':
    Fire(main)