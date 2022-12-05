from dataclasses import dataclass
import os
import json
from functools import partial
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from point_cloud import Pointcloud_DEPRECATED
from abc import ABC


class AbstractViz(ABC):
    class Materials:
        def __init__(self):
            self.pointcloud = rendering.MaterialRecord()  # type: ignore
            self.pointcloud.shader = "defaultUnlit"
            self.pointcloud.point_size = 5.0

            self.line = rendering.MaterialRecord()  # type: ignore
            self.line.shader = "unlitLine"
            self.line.line_width = 1



    def __init__(self):
        print("Building the window...")
        # NOW THE WINDOW
        self._window = gui.Application.instance.create_window("Frament viz", 1600, 1000)  # type: ignore

        self._scene = gui.SceneWidget()  # type: ignore
        self._scene.scene = rendering.Open3DScene(self._window.renderer)  # type: ignore
        self._scene.scene.set_background([1, 1, 1, 1])
        self._scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self._scene.scene.scene.enable_sun_light(True)

        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)  # type: ignore

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self._scene.setup_camera(60, bbox, [0, 0, 0])

        self.materials = self.Materials()


        # self.scene.scene.add_geometry("grid", self.coordinates_grid, grid_material)
        # if grid_dir is not None:
        #     print("Adding grid ...")
        #     with open(grid_dir, 'rb') as f:
        #         grid = np.load(f, allow_pickle=True).item()
        #     self._show_grid(grid, line_material)


        # print("Adding cameras geometry...")
        # for transform_name, geometry in self.camera_geometries.items():
        #     self.scene.scene.add_geometry(f"cam_{transform_name}", geometry, line_material)
        #     self.scene.scene.show_geometry(f"cam_{transform_name}", False)

        # self.pointclouds: Dict[str, Pointcloud_DEPRECATED] = {}

        # for transform_name in transforms_files:
        #     print(f"Adding {transform_name} pointcloud geometry...")
        #     pointcloud = Pointcloud_DEPRECATED.from_dataset(dataset_dir, [transform_name])
        #     self.pointclouds[transform_name] = pointcloud
        #     self.scene.scene.add_geometry(f"pcd_{transform_name}", pointcloud.get_pruned_pointcloud(MAX_POINTCLOUD_POINTS).to_open3d(), self.pcd_material)
        #     self.scene.scene.show_geometry(f"pcd_{transform_name}", False)


        # print("Adding unit cube geometry...")
        # self.scene.scene.add_geometry("unit_cube", self._get_unit_cube_mesh(radius=1.), line_material)
        # print("rendering now!")


        self._window.set_on_layout(self._on_layout)
        self._window.add_child(self._scene)
        self._set_up_settings_panel()


        # We offer handy GUI functions that can affect geometry. This is to check whether requested geometry was valid
        self._valid_geometry = []

    def show_axes(self, show: bool) -> None:
        self._scene.scene.show_axes(show)

    def add_geometry(self, name: str, geometry: o3d.geometry.Geometry, material: Optional[rendering.MaterialRecord] = None) -> None:
        """Add a geometry to the scene
        :param name: name of the geometry
        :param geometry: the geometry to add
        :param material: the material to use for the geometry.
            By default uses self.material.pointcloud for o3d.geometry.Pointcloud, self.material.line otherwise
        """
        if material is None:
            if isinstance(geometry, o3d.geometry.PointCloud):
                material = self.materials.pointcloud
            else:
                material = self.materials.line

        self._scene.scene.add_geometry(name, geometry, material)
        self._valid_geometry.append(name)

    def add_show_checkbox_for_geometry(self, geometry_name: str, panel_section: str, label: str, default_checked: bool = True):
        """Adds checkbox to the corresponding section to show and hide the geometry by name.
        """
        if geometry_name not in self._valid_geometry:
            raise ValueError(f"Unkown geometry '{geometry_name}'")

        def show_geometry(is_checked: bool):
            self._scene.scene.show_geometry(geometry_name, is_checked)

        checkbox = gui.Checkbox(label)  # type: ignore
        checkbox.set_on_checked(show_geometry)
        checkbox.checked = default_checked
        show_geometry(default_checked)

        self.add_settings_panel_child(panel_section, checkbox)

    def add_settings_panel_section(self, panel_section: str, label: str) -> gui.CollapsableVert:
        if panel_section in self._settings_panel_sections:
            raise ValueError(f"Section '{panel_section}' already exists")
        section = gui.CollapsableVert(label, 0.25 * self._settings_em, gui.Margins(self._settings_em, 0, 0, 0))  # type: ignore

        self._settings_panel.add_fixed(self._settings_separation_height)
        self._settings_panel.add_child(section)
        self._settings_panel_sections[panel_section] = section
        return section

    def add_settings_panel_child(self, panel_section: str, child):
        if panel_section not in self._settings_panel_sections:
            raise ValueError(f"Section {panel_section} does not exist yet. Add it with `add_section` method")

        section = self._settings_panel_sections[panel_section]
        assert section is not None

        section.add_child(child)


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

        self._scene.scene.add_geometry("mesh", mesh, line_material)


    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self._window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)  # type: ignore
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)  # type: ignore

    def _set_up_settings_panel(self) -> gui.Vert:  # type: ignore
        em = self._window.theme.font_size
        self._settings_em = em
        self._settings_separation_height = int(round(0.5 * em))
        settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))  # type: ignore

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em, gui.Margins(em, 0, 0, 0))  # type: ignore
        buttons = [
            ("Arcball", gui.SceneWidget.Controls.ROTATE_CAMERA),  # type: ignore
            ("Fly", gui.SceneWidget.Controls.FLY),  # type: ignore
            ("Model", gui.SceneWidget.Controls.ROTATE_MODEL),  # type: ignore
            ("BREAK", None),
            ("Sun", gui.SceneWidget.Controls.ROTATE_IBL),  # type: ignore
            ("Environment", gui.SceneWidget.Controls.ROTATE_IBL),  # type: ignore
        ]

        view_ctrls.add_child(gui.Label("Mouse controls"))  # type: ignore
        row = gui.Horiz(0.25 * em)  # type: ignore
        row.add_stretch()
        for name, camera_mode in buttons:
            if name == "BREAK":
                row.add_stretch()
                view_ctrls.add_child(row)
                row = gui.Horiz(0.25 * em)  # type: ignore
                row.add_stretch()
                continue

            row.add_child(self._get_camera_control_button(name, camera_mode, self._scene))

        row.add_stretch()
        view_ctrls.add_child(row)

        settings_panel.add_fixed(self._settings_separation_height)
        settings_panel.add_child(view_ctrls)
        self._settings_panel = settings_panel
        self._window.add_child(self._settings_panel)
        self._settings_panel_sections = {}



    @staticmethod
    def _get_camera_control_button(name: str, camera_mode, scene):
        button = gui.Button(name)  # type: ignore
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
        self.frame_by_frame_slider_1.set_limits(0, n_frames - 1)
        self.frame_by_frame_slider_1.int_value = 0
        self.frame_by_frame_slider_2.set_limits(0, n_frames - 1)
        self.frame_by_frame_slider_2.int_value = 0

        self._current_slider_pointcloud_idx = idx

        self._on_frame_by_frame_slider_change(0, 'first_slider_object')
        self._on_frame_by_frame_slider_change(0, 'second_slider_object')

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
        self._on_frame_by_frame_slider_change(min(n_frames-1, slider.int_value + 1), object_name)

    def _rewind_slider_pointcloud(self, object_name: str):
        if self._current_slider_pointcloud_idx is None:
            return

        slider = self.frame_by_frame_slider_1 if object_name == 'first_slider_object' else self.frame_by_frame_slider_2
        slider.int_value = max(0, slider.int_value - 1)  # set the value
        self._on_frame_by_frame_slider_change(max(0, slider.int_value - 1), object_name)

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