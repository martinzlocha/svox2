from functools import partial
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from point_cloud import Pointcloud_DEPRECATED, Pointcloud, stack_pointclouds
from fire import Fire
from pointcloud_registration import load_frame_data_from_dataset

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

def load_pointcloud_from_dataset(dataset_dir: str, transforms_file: str) -> Pointcloud:
    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_file)
    pointcloud = stack_pointclouds([frame.pointcloud for frame in frame_data])
    return pointcloud


class VizApplication:
    def __init__(self,
                 dataset_dir: str,
                 grid_dir: Optional[str] = None):
        potential_transforms_files = ["transforms_train.json", "transforms_test.json", "transforms_train_original.json", "transforms_test_original.json", "transforms_train_original_shifted.json", "transforms_train_original_shifted_point_to_point.json", "transforms_train_original_shifted_point_to_plane.json"]
        transforms_files = [f for f in os.listdir(dataset_dir) if f in potential_transforms_files]

        self.transforms = {}
        for transform_file in transforms_files:
            with open(os.path.join(dataset_dir, transform_file)) as f:
                transforms = json.load(f)

            self.transforms[transform_file] = [np.asarray(frame["transform_matrix"]) for frame in transforms["frames"]]

        self.camera_geometries = {}

        print("Building cameras...")
        for transform_name, transforms in self.transforms.items():
            self.camera_geometries[transform_name] = (construct_cameras_geometry(transforms))

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

        self.pcd_material = rendering.MaterialRecord()
        self.pcd_material.shader = "defaultUnlit"
        self.pcd_material.point_size = 5.0

        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 1

        self.scene.scene.show_axes(True)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene.setup_camera(60, bbox, [0, 0, 0])

        em = self.window.theme.font_size


        # self.scene.scene.add_geometry("grid", self.coordinates_grid, grid_material)
        if grid_dir is not None:
            print("Adding grid ...")
            with open(grid_dir, 'rb') as f:
                grid = np.load(f, allow_pickle=True).item()
            self._show_grid(grid, line_material)


        print("Adding cameras geometry...")
        for transform_name, geometry in self.camera_geometries.items():
            self.scene.scene.add_geometry(f"cam_{transform_name}", geometry, line_material)
            self.scene.scene.show_geometry(f"cam_{transform_name}", False)

        self.pointclouds: Dict[str, Pointcloud_DEPRECATED] = {}

        for transform_name in transforms_files:
            print(f"Adding {transform_name} pointcloud geometry...")
            pointcloud = Pointcloud_DEPRECATED.from_dataset(dataset_dir, [transform_name])
            self.pointclouds[transform_name] = pointcloud
            self.scene.scene.add_geometry(f"pcd_{transform_name}", pointcloud.get_pruned_pointcloud(MAX_POINTCLOUD_POINTS).to_open3d(), self.pcd_material)
            self.scene.scene.show_geometry(f"pcd_{transform_name}", False)


        print("Adding unit cube geometry...")
        self.scene.scene.add_geometry("unit_cube", self._get_unit_cube_mesh(radius=1.), line_material)
        print("rendering now!")

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.settings_panel = self._get_settings_panel(em)
        self.window.add_child(self.settings_panel)


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

        self.scene.scene.add_geometry("mesh", mesh, line_material)


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

        frame_by_frame = gui.CollapsableVert("Frame by Frame", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        rb = gui.RadioButton(gui.RadioButton.VERT)
        rb.set_items(list(map(str, self.pointclouds.keys())))
        rb.set_on_selection_changed(self._on_frame_by_frame_select)
        frame_by_frame.add_child(rb)
        self._current_slider_pointcloud_idx = None

        def get_slider(object_name):
            slider_row = gui.Horiz(0.25 * em)
            slider = gui.Slider(gui.Slider.INT)
            slider.set_limits(0, 1)
            slider.set_on_value_changed(partial(self._on_frame_by_frame_slider_change, object_name=object_name))

            minus_button = gui.Button("-")
            minus_button.set_on_clicked(partial(self._rewind_slider_pointcloud, object_name=object_name))

            plus_button = gui.Button("+")
            plus_button.set_on_clicked(partial(self._forward_slider_pointcloud, object_name=object_name))

            slider_row.add_child(minus_button)
            slider_row.add_child(slider)
            slider_row.add_child(plus_button)

            return slider_row, slider

        slider_row_1, self.frame_by_frame_slider_1 = get_slider('first_slider_object')
        slider_row_2, self.frame_by_frame_slider_2 = get_slider('second_slider_object')

        frame_by_frame.add_child(slider_row_1)
        frame_by_frame.add_child(slider_row_2)


        forward_button = gui.Button(">>")
        forward_button.horizontal_padding_em = 0.5
        forward_button.vertical_padding_em = 0
        forward_button.set_on_clicked(self._forward_slider_pointclouds)
        rewind_button = gui.Button("<<")
        rewind_button.horizontal_padding_em = 0.5
        rewind_button.vertical_padding_em = 0
        rewind_button.set_on_clicked(self._rewind_slider_pointclouds)

        row = gui.Horiz(0.25 * em)
        row.add_stretch()
        row.add_child(rewind_button)
        row.add_child(forward_button)
        row.add_stretch()

        frame_by_frame.add_child(row)

        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(frame_by_frame)

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
        self.scene.scene.show_geometry(f"cam_{transform_name}", is_checked)

    def _on_frame_by_frame_select(self, idx):
        pointcloud_key = list(self.pointclouds.keys())[idx]
        pointcloud = self.pointclouds[pointcloud_key]
        n_frames = pointcloud._n_frames
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

        self._on_frame_by_frame_slider_change(first_frame_value, 'first_slider_object')
        self._on_frame_by_frame_slider_change(second_frame_value, 'second_slider_object')

    def _on_frame_by_frame_slider_change(self, value, object_name: str):
        if self._current_slider_pointcloud_idx is None:
            return

        value = int(value)
        pointcloud_key = list(self.pointclouds.keys())[self._current_slider_pointcloud_idx]
        pointcloud = self.pointclouds[pointcloud_key]
        self.scene.scene.remove_geometry(f"{object_name}")

        self.scene.scene.add_geometry(f"{object_name}", pointcloud.from_frame(value).to_open3d(), self.pcd_material)

    def _forward_slider_pointcloud(self, object_name: str):
        if self._current_slider_pointcloud_idx is None:
            return

        pointcloud_key = list(self.pointclouds.keys())[self._current_slider_pointcloud_idx]
        pointcloud = self.pointclouds[pointcloud_key]
        n_frames = pointcloud._n_frames
        assert n_frames is not None

        slider = self.frame_by_frame_slider_1 if object_name == 'first_slider_object' else self.frame_by_frame_slider_2
        slider.int_value = min(n_frames-1, slider.int_value + 1)  # set the value
        self._on_frame_by_frame_slider_change(slider.int_value, object_name)

    def _rewind_slider_pointcloud(self, object_name: str):
        if self._current_slider_pointcloud_idx is None:
            return

        slider = self.frame_by_frame_slider_1 if object_name == 'first_slider_object' else self.frame_by_frame_slider_2
        slider.int_value = max(0, slider.int_value - 1)  # set the value
        self._on_frame_by_frame_slider_change(slider.int_value, object_name)

    def _forward_slider_pointclouds(self):
        self._forward_slider_pointcloud('first_slider_object')
        self._forward_slider_pointcloud('second_slider_object')

    def _rewind_slider_pointclouds(self):
        self._rewind_slider_pointcloud('first_slider_object')
        self._rewind_slider_pointcloud('second_slider_object')

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
