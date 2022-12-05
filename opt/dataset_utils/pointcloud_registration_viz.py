import json
import os
from abstract_viz import AbstractViz
from pointcloud_registration import PairwiseRegistration
from typing import List
from fire import Fire
import open3d.visualization.gui as gui
import numpy as np


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
    def __init__(self, registrations: List[PairwiseRegistration]):
        super().__init__()
        self.pairwise_registrations: List[PairwiseRegistration] = registrations
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
        self._scene.scene.set_geometry_transform('pairwise_registration_pcd_source', registration.transform_matrix)
        self.add_geometry('pairwise_registration_pcd_target', registration.source.pointcloud.as_open3d())

        self.label_source_id.text = f'Source ID: {registration.source.frames[0].frame_data["image_id"]}'
        self.label_target_id.text = f'Target ID: {registration.target.frames[0].frame_data["image_id"]}'
        self.label_type.text = f'Type: {registration.edge_type}'
        self.update_transform_matrix_label(registration.transform_matrix)


def main(dataset_dir: str):
    pairwise_registration_file = os.path.join(dataset_dir, 'pairwise_registrations.json')
    if not os.path.exists(pairwise_registration_file):
        print('Pairwise registration file does not exist: {}'.format(pairwise_registration_file))
        return

    with open(pairwise_registration_file, 'r') as f:
        data = json.load(f)

    pairwise_registrations = [PairwiseRegistration.from_dict(parent_frame_dict) for parent_frame_dict in data]

    gui.Application.instance.initialize()
    viz = RegistrationViz(pairwise_registrations)
    gui.Application.instance.run()

if __name__ == '__main__':
    Fire(main)