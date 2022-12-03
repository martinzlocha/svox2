import math
import os
import shutil
import sys
from typing import Any, Optional, List, Dict
from fire import Fire
import numpy as np
import json
from dataclasses import asdict, dataclass

def do_system(arg):
    print(f"Running command: {arg}")
    err = os.system(arg)
    if err:
        print(f"FATAL: command {arg} failed")
        sys.exit(err)


def to_absolute_path(path: str, parent_dir: Optional[str] = None) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    else:
        parent_dir = os.getcwd() if parent_dir is None else parent_dir
        return os.path.join(parent_dir, path)

@dataclass
class Camera:
    camera_id: str
    view_angle_x: float = 0.  # viewing angle in radians along the x-axis
    view_angle_y: float = 0.  # viewing angle in radians along the y-axis
    fl_x: float = 0.  # focal length along the x-axis
    fl_y: float = 0.  # focal length along the y-axis. In most cameras this will be the same as fl_x
    k1: float = 0.
    k2: float = 0.
    p1: float = 0.
    p2: float = 0.
    cx: float = 0.  # center offset along the x axis. Usually w/2
    cy: float = 0.  # center offset along the y axis. Usually h/2
    w: float = 0.  # width of the image
    h: float = 0.  # height of the image

    @classmethod
    def from_colmap_camera_line(cls, line: str) -> "Camera":
        # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
        # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
        # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
        els = line.split(" ")
        camera_id = f"camera_{els[0]}"
        camera_type = els[1]
        camera = cls(camera_id)

        camera.w = float(els[2])
        camera.h = float(els[3])
        camera.fl_x = float(els[4])
        camera.fl_y = float(els[4])
        # k1 = 0
        # k2 = 0
        # p1 = 0
        # p2 = 0
        camera.cx = camera.w / 2
        camera.cy = camera.h / 2
        if camera_type == "SIMPLE_PINHOLE":
            camera.cx = float(els[5])
            camera.cy = float(els[6])
        elif camera_type == "PINHOLE":
            camera.fl_y = float(els[5])
            camera.cx = float(els[6])
            camera.cy = float(els[7])
        elif camera_type == "SIMPLE_RADIAL":
            camera.cx = float(els[5])
            camera.cy = float(els[6])
            camera.k1 = float(els[7])
        elif camera_type == "RADIAL":
            camera.cx = float(els[5])
            camera.cy = float(els[6])
            camera.k1 = float(els[7])
            camera.k2 = float(els[8])
        elif camera_type == "OPENCV":
            camera.fl_y = float(els[5])
            camera.cx = float(els[6])
            camera.cy = float(els[7])
            camera.k1 = float(els[8])
            camera.k2 = float(els[9])
            camera.p1 = float(els[10])
            camera.p2 = float(els[11])
        else:
            raise ValueError("Unknown camera model ", camera_type)

        # fl = 0.5 * w / tan(0.5 * angle_x);
        camera.view_angle_x = math.atan(camera.w / (camera.fl_x * 2)) * 2
        camera.view_angle_y = math.atan(camera.h / (camera.fl_y * 2)) * 2
        # fovx = angle_x * 180 / math.pi
        # fovy = angle_y * 180 / math.pi

        return camera

    def as_dict(self) -> Dict:
        return asdict(self)


def extract_camera_info(cameras_file: str) -> List[Camera]:
    with open(cameras_file, "r") as f:
        cameras = [Camera.from_colmap_camera_line(line) for line in f if line[0] != "#"]
        return cameras

@dataclass
class Frame:
    image_id: int
    camera_id: str
    file_path: str
    transform_matrix: List[List[float]]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])


def extract_frame_info(images_file: str) -> List[Frame]:
    SKIP_EARLY = 0
    frames: List[Frame] = []

    with open(images_file, "r") as f:
        lines = [line for full_line in f if (line := full_line.strip())[0] != "#"]

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

    for line in lines[SKIP_EARLY*2::2]:
        elems = line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
        camera_id = f"camera_{elems[0]}"
        file_path = "./images/" + '_'.join(elems[9:])
        image_id = int(elems[0])

        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        R = _qvec2rotmat(-qvec)
        t = tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)

        transform_matrix = np.matmul(c2w, flip_mat) # flip cameras (it just works)

        frame = Frame(image_id, camera_id, file_path, transform_matrix.tolist())
        frames.append(frame)

    frames = sorted(frames, key=lambda frame: frame.image_id)

    return frames

def extract_images_from_video(images_dir: str, video_path: str, fps: float) -> None:
    images_dir = to_absolute_path(images_dir)
    video_path = to_absolute_path(video_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File {video_path} not found")

    shutil.rmtree(images_dir, ignore_errors=True)
    os.makedirs(images_dir)

    do_system(f"ffmpeg -i {video_path} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {images_dir}/%04d.jpg")


def run_colmap(images_dir: str, colmap_dir: str, camera_model: str = "OPENCV", matcher_type: str = "exhaustive"):
    db = os.path.join(colmap_dir, "colmap.db")
    images_dir = to_absolute_path(images_dir)
    # db_noext = str(Path(db).with_suffix(""))
    db_noext = os.path.basename(db)
    shutil.rmtree(colmap_dir, ignore_errors=True)
    os.makedirs(colmap_dir)


    text = os.path.join(colmap_dir, "text")
    sparse = os.path.join(colmap_dir, "sparse")
    print(f"running colmap with:\n\tdb={db}\n\timages={images_dir}\n\tsparse={sparse}\n\ttext={text}")
    if os.path.exists(db):
        os.remove(db)

    # FEATURE EXTRACTOR
    do_system(f"colmap feature_extractor --ImageReader.camera_model {camera_model} --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images_dir}")

    # MATCHER
    # exhaustive_matcher, vocab_tree_matcher, sequential_matcher, spatial_matcher, transitive_matcher, matches_importer
    do_system(f"colmap {matcher_type}_matcher --SiftMatching.guided_matching=true --database_path {db}")
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")

    # MAPPER
    do_system(f"colmap mapper --database_path {db} --image_path {images_dir} --output_path {sparse}")

    # BUNDLE ADJUSTER
    do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")


def generate_camera_transforms(colmap_dir: str, keep_colmap_coords: bool = True) -> None:
    colmap_dir = to_absolute_path(colmap_dir)
    cameras_file = os.path.join(colmap_dir, "text", "cameras.txt")
    images_file = os.path.join(colmap_dir, "text", "images.txt")
    output_file = os.path.join(colmap_dir, "transforms.json")

    cameras = extract_camera_info(cameras_file)
    frames = extract_frame_info(images_file)
    out = {
        "cameras": [camera.as_dict() for camera in cameras],
        "frames": [frame.as_dict() for frame in frames],
        "camera_angle_x": 0.9311120127176192,  # TODO: read from dataset
    }

    out.update(cameras[0].as_dict())  # for backward compatibility

    print(f"writing {output_file}")
    with open(output_file, "w") as outfile:
        json.dump(out, outfile, indent=2)


def copy_images(dataset_dir: str, images_dir: str) -> str:
    dataset_dir = to_absolute_path(dataset_dir)
    images_dir = to_absolute_path(images_dir)
    new_images_dir = os.path.join(dataset_dir, "images")

    if new_images_dir == images_dir:
        return images_dir

    shutil.rmtree(new_images_dir, ignore_errors=True)
    os.makedirs(new_images_dir)
    # print(f"cp {images_dir}/* {new_images_dir}/")
    do_system(f"cp {images_dir}/* {new_images_dir}/")

    return new_images_dir


def main(dataset_dir: str, images_dir: Optional[str] = None, video_path: Optional[str] = None, fps: float = 1.0, matcher_type: str = "exhaustive"):
    dataset_dir = to_absolute_path(dataset_dir)

    if images_dir is None:
        images_dir = os.path.join(dataset_dir, "images")
        if video_path is None:
            raise ValueError("either images dir or video path has to be specified")
        extract_images_from_video(images_dir, video_path, fps)

    colmap_dir = os.path.join(dataset_dir, "colmap")
    images_dir = copy_images(dataset_dir, images_dir)

    run_colmap(images_dir, colmap_dir, matcher_type=matcher_type)
    generate_camera_transforms(colmap_dir)


if __name__ == "__main__":
    Fire(main)