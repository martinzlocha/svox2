from functools import partial
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from point_cloud import Pointcloud, stack_pointclouds
from fire import Fire
from abstract_viz import AbstractViz, shift_int_slider
from pointcloud_registration import load_frame_data_from_dataset
import sys
print(sys.path)

MAX_POINTCLOUD_POINTS = 1000000

def to_absolute_path(path: str, parent_dir: Optional[str] = None) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    else:
        parent_dir = os.getcwd() if parent_dir is None else parent_dir
        return os.path.join(parent_dir, path)


def construct_cameras_geometry(transforms: List[np.ndarray]) -> o3d.geometry.LineSet:
    w: float = 0.19
    h: float = 0.1
    length = w * 2
    vertices = []
    lines = []

    for i, transform in enumerate(transforms):
        camera_vertices = np.array([
            [ 0,    0,    0,  1],
            [ w,  h, -length, 1], # top right
            [ w, -h, -length, 1], # bottom right
            [-w, -h, -length, 1], # bottom left
            [-w,  h, -length, 1], # top left
            [ 0, 1.5 * h, -length, 1], # corner
        ])
        camera_vertices = camera_vertices @ transform.T
        camera_lines = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [5, 1],
            [5, 4],
        ])
        camera_lines = camera_lines + len(vertices) * 6

        vertices.append(camera_vertices)
        lines.append(camera_lines)

    vertices = np.concatenate(vertices, axis=0)
    lines = np.concatenate(lines, axis=0)

    geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices[:, :3]), o3d.utility.Vector2iVector(lines))

    # generate 3 random numbers between 0 and 1
    color = np.random.rand(3)
    geometry.paint_uniform_color(color)

    return geometry

def load_pointcloud_from_dataset(dataset_dir: str, transforms_file: str) -> List[Pointcloud]:
    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_file)
    return [frame.pointcloud for frame in frame_data]


class VizApplication(AbstractViz):
    def __init__(self,
                 dataset_dir: str,
                 grid_dir: Optional[str] = None):
        super().__init__()
        potential_transforms_files = ["transforms_train.json", "transforms_test.json", "transforms_train_original.json", "transforms_test_original.json", "transforms_train_original_shifted.json"]
        transforms_files = [f for f in os.listdir(dataset_dir) if f in potential_transforms_files]

        self.transforms = {}
        for transform_file in transforms_files:
            with open(os.path.join(dataset_dir, transform_file)) as f:
                transforms = json.load(f)

            self.transforms[transform_file] = [np.asarray(frame["transform_matrix"]) for frame in transforms["frames"]]

        if grid_dir is not None:
            print("Adding grid ...")
            with open(grid_dir, 'rb') as f:
                grid = np.load(f, allow_pickle=True).item()
            self._show_grid(grid, self.materials.line)

        self._add_cameras()
        self._add_pointclouds(dataset_dir, transforms_files)
        self._add_unit_cube()

        self._add_to_settings_panel()

        print("rendering now!")

    def _add_cameras(self, ):
        self.camera_geometries = {}
        self.add_settings_panel_section("cameras_checkbox_section", "Show Cameras")

        print("Building cameras...")
        for transform_name, transforms in self.transforms.items():
            self.camera_geometries[transform_name] = (construct_cameras_geometry(transforms))

        print("Adding cameras geometry...")
        for transform_name, geometry in self.camera_geometries.items():
            self.add_geometry(f"cam_{transform_name}", geometry, self.materials.line)
            self.add_show_checkbox_for_geometry(f"cam_{transform_name}", "cameras_checkbox_section", transform_name, False)

    def _add_pointclouds(self, dataset_dir: str, transforms_files: Iterable[str]) -> None:
        self.pointclouds: Dict[str, List[Pointcloud]] = {}
        self.add_settings_panel_section("pointclouds_checkbox_section", "Show Pointcloud")
        for transform_name in transforms_files:
            print(f"Adding {transform_name} pointcloud geometry...")
            # pointcloud = Pointcloud_DEPRECATED.from_dataset(dataset_dir, [transform_name])
            pointcloud = load_pointcloud_from_dataset(dataset_dir, transform_name)
            self.pointclouds[transform_name] = pointcloud
            open3d_pointcloud = stack_pointclouds(pointcloud).prune(MAX_POINTCLOUD_POINTS).as_open3d()
            self.add_geometry(f"pcd_{transform_name}", open3d_pointcloud, self.materials.pointcloud)
            self.add_show_checkbox_for_geometry(f"pcd_{transform_name}", "pointclouds_checkbox_section", transform_name, False)

    def _add_unit_cube(self):
        print("Adding unit cube geometry...")

        self.add_settings_panel_section("unit_cube", "Unit cube")
        self.add_geometry("unit_cube", self._get_unit_cube_mesh(radius=1.), self.materials.line)
        self.add_show_checkbox_for_geometry("unit_cube", "unit_cube", "unit cube", False)


    def _show_grid(self, grid: Dict[str, np.ndarray], line_material) -> None:
        side_radius = grid['side_length'] / 2
        mesh = o3d.geometry.LineSet()
        print('sum of den', grid['densities'].min())
        mask = (grid['densities'] > 0.5)[..., 0]
        locations = grid['locations'][mask]
        print(locations.shape)
        cube_edges = np.array([
                [side_radius, side_radius, side_radius],
                [side_radius, side_radius, -side_radius],
                [side_radius, -side_radius, side_radius],
                [side_radius, -side_radius, -side_radius],
                [-side_radius, side_radius, side_radius],
                [-side_radius, side_radius, -side_radius],
                [-side_radius, -side_radius, side_radius],
                [-side_radius, -side_radius, -side_radius],
            ])
        cube_lines = np.array([
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ])


        location_count = locations.shape[0]
        locations_repeated = np.repeat(locations, 8, 0)
        cube_edges = np.tile(cube_edges, [location_count, 1])
        cube_edges += locations_repeated
        location_increments = np.arange(location_count)[..., None]
        location_increments = np.repeat(location_increments, 12, 0) * 8
        cube_lines = np.tile(cube_lines, [location_count, 1])
        cube_lines += location_increments

        # get vertices of a unit cube with center at origin
        mesh.points = o3d.utility.Vector3dVector(cube_edges)
        mesh.lines = o3d.utility.Vector2iVector(cube_lines)
        mesh.paint_uniform_color((0., 0., 1.))

        self.add_geometry("grid", mesh, line_material)

        self.add_settings_panel_section("grid_checkbox_section", "Show Grid")
        self.add_show_checkbox_for_geometry("grid", "grid_checkbox_section", "Show Grid", False)
        # self._scene.scene.add_geometry("mesh", mesh, line_material)

    def _add_to_settings_panel(self):
        # TODO: move to the abstract viz
        em = self._settings_em
        frame_by_frame = self.add_settings_panel_section("replay_capture", "Replay Capture")

        rb = gui.RadioButton(gui.RadioButton.VERT)
        rb.set_items(list(map(str, self.pointclouds.keys())))
        rb.set_on_selection_changed(self._on_frame_by_frame_select)
        # frame_by_frame.add_child(rb)
        self.add_settings_panel_child("replay_capture", rb)
        self._current_slider_pointcloud_idx = None

        self.frame_by_frame_slider_1 = self.add_slider_selection("replay_capture", (0, 1), partial(self._on_frame_by_frame_slider_change, object_name='first_slider_object'))
        self.frame_by_frame_slider_2 = self.add_slider_selection("replay_capture", (0, 1), partial(self._on_frame_by_frame_slider_change, object_name='second_slider_object'))


        def shift_sliders(value: int):
            shift_int_slider(self.frame_by_frame_slider_1, value, partial(self._on_frame_by_frame_slider_change, object_name='first_slider_object'))
            shift_int_slider(self.frame_by_frame_slider_2, value, partial(self._on_frame_by_frame_slider_change, object_name='second_slider_object'))


        forward_button = gui.Button(">>")
        forward_button.horizontal_padding_em = 0.5
        forward_button.vertical_padding_em = 0
        forward_button.set_on_clicked(partial(shift_sliders, 1))
        rewind_button = gui.Button("<<")
        rewind_button.horizontal_padding_em = 0.5
        rewind_button.vertical_padding_em = 0
        rewind_button.set_on_clicked(partial(shift_sliders, -1))

        row = gui.Horiz(0.25 * em)
        row.add_stretch()
        row.add_child(rewind_button)
        row.add_child(forward_button)
        row.add_stretch()

        # frame_by_frame.add_child(row)
        self.add_settings_panel_child("replay_capture", row)

    @staticmethod
    def _get_unit_cube_mesh(radius: float = 0.5,
                            translation: Optional[Tuple[float, float, float]] = None,
                            colour: Tuple[float, float, float] = (1., 0., 0.)):
        unit_cube = o3d.geometry.LineSet()
        cube_edges = [
            [radius, radius, radius],
            [radius, radius, -radius],
            [radius, -radius, radius],
            [radius, -radius, -radius],
            [-radius, radius, radius],
            [-radius, radius, -radius],
            [-radius, -radius, radius],
            [-radius, -radius, -radius],
        ]

        if translation is not None:
            translated_edges = []
            for (ex, ey, ez) in cube_edges:
                translated_edges.append([ex + translation[0],
                                         ey + translation[1],
                                         ez + translation[2]])
            cube_edges = translated_edges


        # get vertices of a unit cube with center at origin
        unit_cube.points = o3d.utility.Vector3dVector(cube_edges)

        unit_cube.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ])
        unit_cube.paint_uniform_color(colour)

        return unit_cube

    def _on_frame_by_frame_select(self, idx):
        pointcloud_key = list(self.pointclouds.keys())[idx]
        pointcloud = self.pointclouds[pointcloud_key]
        n_frames = len(pointcloud)
        assert n_frames is not None
        first_frame_value = min(self.frame_by_frame_slider_1.int_value, n_frames-1)
        second_frame_value = min(self.frame_by_frame_slider_2.int_value, n_frames-1)

        self.frame_by_frame_slider_1.set_limits(0, n_frames - 1)
        # self.frame_by_frame_slider_1.int_value = 0
        self.frame_by_frame_slider_2.set_limits(0, n_frames - 1)
        # self.frame_by_frame_slider_2.int_value = 0

        self._current_slider_pointcloud_idx = idx

        self.frame_by_frame_slider_1.int_value = first_frame_value
        self.frame_by_frame_slider_2.int_value = second_frame_value

        self._on_frame_by_frame_slider_change(self.frame_by_frame_slider_1, first_frame_value, 'first_slider_object')
        self._on_frame_by_frame_slider_change(self.frame_by_frame_slider_2, second_frame_value, 'second_slider_object')

    def _on_frame_by_frame_slider_change(self, slider, value, object_name: str):
        if self._current_slider_pointcloud_idx is None:
            return

        value = int(value)
        pointcloud_key = list(self.pointclouds.keys())[self._current_slider_pointcloud_idx]
        pointcloud = self.pointclouds[pointcloud_key]
        self._scene.scene.remove_geometry(f"{object_name}")

        self._scene.scene.add_geometry(f"{object_name}", pointcloud[value].as_open3d(), self.materials.pointcloud)

    def _forward_slider_pointclouds(self):
        self._forward_slider_pointcloud('first_slider_object')
        self._forward_slider_pointcloud('second_slider_object')

    def _rewind_slider_pointclouds(self):
        self._rewind_slider_pointcloud('first_slider_object')
        self._rewind_slider_pointcloud('second_slider_object')


def main(dataset_dir: str,
         grid_dir: Optional[str] = None):
    dataset_dir = to_absolute_path(dataset_dir)

    gui.Application.instance.initialize()
    VizApplication(dataset_dir, grid_dir)
    gui.Application.instance.run()

if __name__ == "__main__":
    Fire(main)
