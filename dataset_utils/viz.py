from functools import partial
import json
from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from point_cloud import Pointcloud
import torch
from fire import Fire

MAX_POINTCLOUD_POINTS = 1000000

def to_absolute_path(path: str, parent_dir: Optional[str] = None) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    else:
        parent_dir = os.getcwd() if parent_dir is None else parent_dir
        return os.path.join(parent_dir, path)


def get_o3d_pointcloud(pointcloud: Pointcloud) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points.numpy())

    features = pointcloud.features
    if features is None:
        features = np.ones((pointcloud.points.shape[0], 3))
    else:
        features = features.numpy() / 255

    pcd.colors = o3d.utility.Vector3dVector(features)
    return pcd


class Camera:
    def __init__(self, name: str, camera_transform: np.ndarray, w: float = 0.19, h: float = 0.1, s: float=0.0):
        self.name = name
        self.camera_transform = camera_transform
        length = w * 2

        vertices = np.array([
            [ 0,    0,    0,  1],
            [ w,  h, -length, 1], # top right
            [ w, -h, -length, 1], # bottom right
            [-w, -h, -length, 1], # bottom left
            [-w,  h, -length, 1], # top left
            [ 0, 1.5 * h, -length, 1], # corner
        ])

        lines = np.array([
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

        vertices = vertices @ camera_transform.T

        self.geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices[:, :3]), o3d.utility.Vector2iVector(lines))
        # self.geometry.transform(self.camera_transform)


        self.geometry.paint_uniform_color((0.0, 0.0, 1.0))

    def get_complete_transform_matrix(self, matrix):
        return matrix @ self.camera_transform


class VizApplication:
    def __init__(self,
                 dataset_dir: str,
                 grid_dir: Optional[str] = None):
        potential_transforms_files = ["transforms_train.json", "transforms_test.json", "transforms_train_original.json", "transforms_test_original.json"]
        transforms_files = [f for f in os.listdir(dataset_dir) if f in potential_transforms_files]

        self.transforms = {}
        for transform_file in transforms_files:
            with open(os.path.join(dataset_dir, transform_file)) as f:
                transforms = json.load(f)

            self.transforms[transform_file] = [np.asarray(frame["transform_matrix"]) for frame in transforms["frames"]]

        self.cameras = {}

        print("Building cameras...")
        for transform_name, transforms in self.transforms.items():
            self.cameras[transform_name] = []
            for i, transform in enumerate(transforms):
                camera = Camera(f"camera_{i}_{transform_name}", transform)
                self.cameras[transform_name].append(camera)

        print("Building the window...")
        # NOW THE WINDOW
        self.window = gui.Application.instance.create_window(
            "Colmap viz", 1600, 1000)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_sun_light(True)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"

        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 1

        self.scene.scene.show_axes(True)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene.setup_camera(60, bbox, [0, 0, 0])

        em = self.window.theme.font_size
        self.settings_panel = self._get_settings_panel(em)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.settings_panel)

        # self.scene.scene.add_geometry("grid", self.coordinates_grid, grid_material)
        if grid_dir is not None:
            print("Adding grid ...")
            with open(grid_dir, 'rb') as f:
                grid = np.load(f, allow_pickle=True).item()
            self._show_grid(grid, line_material=line_material)


        print("Adding cameras geometry...")
        for transform_name, cameras in self.cameras.items():
            for camera in cameras:
                self.scene.scene.add_geometry(camera.name, camera.geometry, line_material)
                self.scene.scene.show_geometry(camera.name, False)

        for transform_name in transforms_files:
            print(f"Adding {transform_name} pointcloud geometry...")
            pointcloud = Pointcloud.from_dataset(dataset_dir, [transform_name], translation=translation, scaling=scaling)
            self.scene.scene.add_geometry(f"pcd_{transform_name}", get_o3d_pointcloud(pointcloud.get_pruned_pointcloud(MAX_POINTCLOUD_POINTS)), material)
            self.scene.scene.show_geometry(f"pcd_{transform_name}", False)

        print("Adding unit cube geometry...")
        self.scene.scene.add_geometry("unit_cube", self._get_unit_cube_mesh(radius=1.), line_material)
        print("rendering now!")

    def _show_grid(self, grid: Dict[str, np.array], line_material) -> None:
        side_length = grid['side_length']
        to_3_tuple = lambda x_arr: (float(x_arr[0]), float(x_arr[1]), float(x_arr[2]))
        # Grid has locations, cube sizes, cube colors
        for i, location in enumerate(grid['locations']):
            self.scene.scene.add_geometry(f"grid_cube_{i}",
                                          self._get_unit_cube_mesh(side_length / 2,
                                                                   to_3_tuple(location)),
                                          line_material)


    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self.scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self.settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _get_settings_panel(self, em: float):
        separation_height = int(round(0.5 * em))
        settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        buttons = [
            ("Arcball", gui.SceneWidget.Controls.ROTATE_CAMERA),
            ("Fly", gui.SceneWidget.Controls.FLY),
            ("Model", gui.SceneWidget.Controls.ROTATE_MODEL),
            ("BREAK", None),
            ("Sun", gui.SceneWidget.Controls.ROTATE_IBL),
            ("Environment", gui.SceneWidget.Controls.ROTATE_IBL),
        ]

        view_ctrls.add_child(gui.Label("Mouse controls"))
        row = gui.Horiz(0.25 * em)
        row.add_stretch()
        for name, camera_mode in buttons:
            if name == "BREAK":
                row.add_stretch()
                view_ctrls.add_child(row)
                row = gui.Horiz(0.25 * em)
                row.add_stretch()
                continue

            row.add_child(self._get_camera_control_button(name, camera_mode, self.scene))

        row.add_stretch()
        view_ctrls.add_child(row)

        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(view_ctrls)

        dataset_selection = gui.CollapsableVert("Show cameras", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        for transform_name in self.transforms.keys():
            cb = gui.Checkbox(transform_name)
            cb.set_on_checked(partial(self._on_check, transform_name=transform_name))
            dataset_selection.add_child(cb)


        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(dataset_selection)

        pointcloud_selection = gui.CollapsableVert("Show pointcloud", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        for transform_name in self.transforms.keys():
            cb = gui.Checkbox(transform_name)
            cb.set_on_checked(partial(self._on_check_pcd, transform_name=transform_name))
            pointcloud_selection.add_child(cb)


        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(pointcloud_selection)

        unit_cube = gui.CollapsableVert("Unit Cube", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        cb = gui.Checkbox("Show unit cube")
        cb.set_on_checked(self._on_check_unit_cube)
        cb.checked = True
        unit_cube.add_child(cb)

        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(unit_cube)

        settings_panel.visible = True

        return settings_panel

    @staticmethod
    def _get_camera_control_button(name: str, camera_mode, scene):
        button = gui.Button(name)
        button.horizontal_padding_em = 0.5
        button.vertical_padding_em = 0
        button.set_on_clicked(lambda: scene.set_view_controls(camera_mode))
        return button

    @staticmethod
    def _get_unit_cube_mesh(radius: int = 0.5,
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

    def _on_check(self, is_checked: bool, transform_name: str):
        cameras = self.cameras[transform_name]
        for camera in cameras:
            self.scene.scene.show_geometry(camera.name, is_checked)

    def _on_check_pcd(self, is_checked: bool, transform_name: str):
        self.scene.scene.show_geometry(f"pcd_{transform_name}", is_checked)

        print(transform_name, is_checked)

    def _on_check_unit_cube(self, is_checked: bool):
        self.scene.scene.show_geometry("unit_cube", is_checked)


def main(dataset_dir: str,
         grid_dir: Optional[str] = None):
    dataset_dir = to_absolute_path(dataset_dir)

    gui.Application.instance.initialize()
    VizApplication(dataset_dir, grid_dir)
    gui.Application.instance.run()

if __name__ == "__main__":
    Fire(main)
