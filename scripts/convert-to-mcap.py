import math
import cv2
import struct
import numpy as np
import open3d as o3d
from mcap_protobuf.writer import Writer as McapWriter
from simplejpeg import encode_jpeg, decode_jpeg
from pathlib import Path
from argparse import ArgumentParser
from loguru import logger
from truckscenes import TruckScenes
from tqdm import tqdm
import polars as pl
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from google.protobuf.timestamp_pb2 import Timestamp
from pyproj import Transformer
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value, BoolValue, StringValue


def quat_to_heading(quat):
    q_x, q_y, q_z, q_w = quat
    # Compute yaw (heading) in radians
    yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))

    # Convert yaw to degrees
    yaw_degrees = math.degrees(yaw)

    # Normalize heading to [0, 360) degrees
    heading = yaw_degrees % 360

    return heading


# Western Europe
transformer_32N = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


CAMERAS = [
    "CAMERA_LEFT_FRONT",
    "CAMERA_LEFT_BACK",
    "CAMERA_RIGHT_FRONT",
    "CAMERA_RIGHT_BACK",
]
RADARS = [
    "RADAR_RIGHT_BACK",
    "RADAR_RIGHT_SIDE",
    "RADAR_RIGHT_FRONT",
    "RADAR_LEFT_FRONT",
    "RADAR_LEFT_SIDE",
    "RADAR_LEFT_BACK",
]
LIDARS = [
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
    "LIDAR_TOP_FRONT",
    "LIDAR_TOP_LEFT",
    "LIDAR_TOP_RIGHT",
    "LIDAR_REAR",
]
IMU = ["XSENSE_CABIN", "XSENSE_CHASSIS"]
EGO_MOTION = [
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "yaw",
    "pitch",
    "roll",
    "yaw_rate",
    "pitch_rate",
    "roll_rate",
]


def convert(rootdir, split, dest) -> None:
    trucksc = TruckScenes(f"v1.0-{split}", rootdir, True)
    calibrations = {
        calib["token"]: {
            "rotation": calib["rotation"],
            "translation": calib["translation"],
        }
        for calib in trucksc.calibrated_sensor
    }
    df_ego_pose = pl.DataFrame(trucksc.ego_pose)
    df_ego_motion_cabin = pl.DataFrame(trucksc.ego_motion_cabin)
    df_ego_motion_chassis = pl.DataFrame(trucksc.ego_motion_chassis)

    scene_bar = tqdm(trucksc.scene, ascii=True)
    for scene in scene_bar:

        filename = Path(f"{dest}/v1.0-{split}/{scene['name']}/episode.mcap")
        filename.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing {filename}")
        scene_bar.set_description(f"scene: {scene['name']}")

        # get first and last sample to get timestamps
        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]

        begin_sample = trucksc.get("sample", first_sample_token)
        begin_timestamp = begin_sample["timestamp"]
        end_sample = trucksc.get("sample", last_sample_token)
        end_timestamp = end_sample["timestamp"]

        ts = Timestamp()
        with filename.open("wb") as pfile:
            writer = McapWriter(pfile)

            ts.FromMicroseconds(int(begin_timestamp))
            msg = StringValue(value=scene["name"])
            writer.write_message(
                topic="/info/name",
                log_time=ts.ToNanoseconds(),
                message=msg,
                publish_time=ts.ToNanoseconds(),
            )

            msg = Int32Value(value=scene["nbr_samples"])
            writer.write_message(
                topic="/info/nbr_samples",
                log_time=ts.ToNanoseconds(),
                message=msg,
                publish_time=ts.ToNanoseconds(),
            )

            for desc in scene["description"].split(";"):
                name, value = desc.split(".")
                writer.write_message(
                    topic=f"/info/{name}",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

            # pose
            df_scene_ego_pose = df_ego_pose.filter(
                pl.col("timestamp") >= begin_timestamp
            ).filter(pl.col("timestamp") <= end_timestamp)
            pbar = tqdm(
                df_scene_ego_pose.iter_rows(named=True),
                total=df_scene_ego_pose.shape[0],
                ascii=True,
            )
            for pose in pbar:
                pbar.set_description(f"pose:{pose['token']}")
                timestamp = pose["timestamp"]

                ts.FromMicroseconds(int(timestamp))
                easting, northing, altitude = pose["translation"]
                heading = quat_to_heading(pose["rotation"])
                longitude, latitude = transformer_32N.transform(easting, northing)
                gnss = LocationFix(
                    timestamp=ts,
                    frame_id="ego_pose",
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                )
                writer.write_message(
                    topic="/sensor/ego_pose/location",
                    log_time=ts.ToNanoseconds(),
                    message=gnss,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=heading)
                writer.write_message(
                    topic="/sensor/ego_pose/orientation",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )
            # cabin motion
            df_scene_ego_motion_cabin = df_ego_motion_cabin.filter(
                pl.col("timestamp") >= begin_timestamp
            ).filter(pl.col("timestamp") <= end_timestamp)
            pbar = tqdm(
                df_scene_ego_motion_cabin.iter_rows(named=True),
                total=df_scene_ego_motion_cabin.shape[0],
                ascii=True,
            )
            for motion in pbar:
                pbar.set_description(f"ego motion cabin:{motion['token']}")
                timestamp = motion["timestamp"]
                ts = Timestamp()
                ts.FromMicroseconds(int(timestamp))
                for key in EGO_MOTION:
                    msg = FloatValue(value=motion[key])
                    writer.write_message(
                        topic=f"/sensor/ego_motion_cabin/{key}",
                        log_time=ts.ToNanoseconds(),
                        message=msg,
                        publish_time=ts.ToNanoseconds(),
                    )

            # chassis motion
            df_scene_ego_motion_chassis = df_ego_motion_chassis.filter(
                pl.col("timestamp") >= begin_timestamp
            ).filter(pl.col("timestamp") <= end_timestamp)
            pbar = tqdm(
                df_scene_ego_motion_chassis.iter_rows(named=True),
                total=df_scene_ego_motion_chassis.shape[0],
                ascii=True,
            )
            for motion in pbar:
                pbar.set_description(f"ego motion chassis:{motion['token']}")
                timestamp = motion["timestamp"]
                ts = Timestamp()
                ts.FromMicroseconds(int(timestamp))
                for key in EGO_MOTION:
                    msg = FloatValue(value=motion[key])
                    writer.write_message(
                        topic=f"/sensor/ego_motion_chassis/{key}",
                        log_time=ts.ToNanoseconds(),
                        message=msg,
                        publish_time=ts.ToNanoseconds(),
                    )

            # sensor data
            sample_token = scene["first_sample_token"]
            sample_bar = tqdm(range(scene["nbr_samples"]), ascii=True)

            for _ in sample_bar:

                sample = trucksc.get("sample", sample_token)
                data = sample["data"]
                timestamp = sample["timestamp"]
                ts = Timestamp()
                ts.FromMicroseconds(int(timestamp))

                for sensor in CAMERAS:
                    sample_bar.set_description(f"{sensor:16s}:{sample_token}")
                    cam = trucksc.get("sample_data", data[sensor])
                    image_file = f"{trucksc.dataroot}/{cam['filename']}"
                    with Path(image_file).open("rb") as file:
                        bytes = file.read()

                    img = CompressedImage(
                        timestamp=ts,
                        format="jpeg",
                        data=bytes,
                    )

                    writer.write_message(
                        topic=f"/sensor/{sensor}",
                        log_time=ts.ToNanoseconds(),
                        publish_time=ts.ToNanoseconds(),
                        message=img,
                    )

                for sensor in RADARS:
                    sample_bar.set_description(f"{sensor:16s}:{sample_token}")
                    radar = trucksc.get("sample_data", data[sensor])
                    pcd_file = f"{trucksc.dataroot}/{radar['filename']}"
                    pcd = o3d.io.read_point_cloud(pcd_file)
                    calib = calibrations[radar["calibrated_sensor_token"]]
                    x, y, z = calib["translation"]
                    position = Vector3(x=x, y=y, z=z)
                    qw, qx, qy, qz = calib["rotation"]
                    orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                    pose = Pose(position=position, orientation=orientation)

                    # Define the fields of the point cloud
                    fields = [
                        PackedElementField(
                            name="x", offset=0, type=PackedElementField.FLOAT32
                        ),
                        PackedElementField(
                            name="y", offset=4, type=PackedElementField.FLOAT32
                        ),
                        PackedElementField(
                            name="z", offset=8, type=PackedElementField.FLOAT32
                        ),
                    ]

                    # Pack point data into binary format
                    point_stride = 12  # Each point is 3 floats (x, y, z), 4 bytes each
                    points = np.asarray(pcd.points, dtype=np.float32)

                    # Construct the PointCloud message
                    point_cloud_msg = PointCloud(
                        timestamp=ts,
                        frame_id="ego",
                        pose=pose,
                        point_stride=point_stride,
                        fields=fields,
                        data=points.tobytes(),
                    )

                    writer.write_message(
                        topic=f"/sensor/{sensor}",
                        log_time=ts.ToNanoseconds(),
                        publish_time=ts.ToNanoseconds(),
                        message=point_cloud_msg,
                    )

                for sensor in LIDARS:
                    sample_bar.set_description(f"{sensor:16s}:{sample_token}")
                    radar = trucksc.get("sample_data", data[sensor])
                    pcd_file = f"{trucksc.dataroot}/{radar['filename']}"
                    pcd = o3d.io.read_point_cloud(pcd_file)
                    calib = calibrations[radar["calibrated_sensor_token"]]
                    x, y, z = calib["translation"]
                    position = Vector3(x=x, y=y, z=z)
                    qw, qx, qy, qz = calib["rotation"]
                    orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                    pose = Pose(position=position, orientation=orientation)

                    # Define the fields of the point cloud
                    fields = [
                        PackedElementField(
                            name="x", offset=0, type=PackedElementField.FLOAT32
                        ),
                        PackedElementField(
                            name="y", offset=4, type=PackedElementField.FLOAT32
                        ),
                        PackedElementField(
                            name="z", offset=8, type=PackedElementField.FLOAT32
                        ),
                    ]

                    # Pack point data into binary format
                    point_stride = 12  # Each point is 3 floats (x, y, z), 4 bytes each
                    points = np.asarray(pcd.points, dtype=np.float32)

                    # Construct the PointCloud message
                    point_cloud_msg = PointCloud(
                        timestamp=ts,
                        frame_id="ego",
                        pose=pose,
                        point_stride=point_stride,
                        fields=fields,
                        data=points.tobytes(),
                    )

                    writer.write_message(
                        topic=f"/sensor/{sensor}",
                        log_time=ts.ToNanoseconds(),
                        publish_time=ts.ToNanoseconds(),
                        message=point_cloud_msg,
                    )

                sample_token = sample["next"]

            writer.finish()


if __name__ == "__main__":

    parser = ArgumentParser("Convert MAN truckschenes to mcap for Nutron")
    parser.add_argument(
        "-s",
        "--split",
        choices=["trainval", "test", "mini"],
        help="Which split to convert",
        default="mini",
    )
    parser.add_argument(
        "-r",
        "--root",
        help="Which split to convert",
        default="/nasa/3rd_party/man/man-truckscenes",
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="Destination dir",
        default="/nasa/3rd_party/man/man-truckscenes-mcap",
    )

    args = parser.parse_args()

    convert(args.root, args.split, args.dest)
