"""Build a standalone MuJoCo scene from DIAN_rl_compressed.urdf.

Visual and collision meshes are copied from the paths specified by the URDF.
MuJoCo uses convex hulls for mesh collisions on moving bodies.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/home/hi/下载/点足总装修订版10-18/DIAN")
DEFAULT_ANGLES = {
    "FL_J1": 0.0, "FL_J2": 0.9, "FL_J3": -0.5,
    "FR_J1": 0.0, "FR_J2": -0.9, "FR_J3": 0.5,
    "RL_J1": 0.0, "RL_J2": 0.7, "RL_J3": -0.5,
    "RR_J1": 0.0, "RR_J2": -0.7, "RR_J3": 0.5,
}


def origin_attributes(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    result = {}
    if element.get("xyz"):
        result["pos"] = element.get("xyz")
    if element.get("rpy") and element.get("rpy") != "0 0 0":
        result["euler"] = element.get("rpy")
    return result


def copy_mesh(mesh_file: Path, output: Path) -> Path:
    """Keep all STL triangles, converting oversized binary STL files to OBJ."""
    with mesh_file.open("rb") as source_file:
        header = source_file.read(84)
    if len(header) != 84:
        raise ValueError(f"Invalid binary STL: {mesh_file}")
    face_count = struct.unpack("<I", header[80:84])[0]
    if mesh_file.stat().st_size != 84 + 50 * face_count:
        raise ValueError(f"Expected binary STL: {mesh_file}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if face_count <= 200_000:
        shutil.copy2(mesh_file, output)
        return output

    triangle_type = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    with mesh_file.open("rb") as source_file:
        source_file.seek(84)
        triangles = np.frombuffer(source_file.read(), dtype=triangle_type)["vertices"].reshape(-1, 3)
    vertices, inverse = np.unique(triangles, axis=0, return_inverse=True)
    output = output.with_suffix(".obj")
    with output.open("w", encoding="ascii") as target:
        for vertex in vertices:
            target.write("v %.9g %.9g %.9g\n" % tuple(vertex))
        for face in inverse.reshape(-1, 3) + 1:
            target.write("f %d %d %d\n" % tuple(face))
    return output


def build(source: Path, destination: Path) -> None:
    urdf_file = source / "urdf" / "DIAN_rl_compressed.urdf"
    urdf = ET.parse(urdf_file).getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_names = set()
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children[parent].append(joint)
        child_names.add(child)
    roots = set(links) - child_names
    if roots != {"base_link"}:
        raise ValueError(f"Expected base_link as sole URDF root, got {roots}")

    mj = ET.Element("mujoco", model="mydog_DIAN")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="assets", autolimits="true")
    ET.SubElement(mj, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast")
    assets = ET.SubElement(mj, "asset")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.55 0.7 0.85", rgb2="0.95 0.96 0.98", width="512", height="3072")
    ET.SubElement(assets, "texture", name="ground_checker", type="2d", builtin="checker",
                  rgb1="0.38 0.43 0.47", rgb2="0.68 0.72 0.75", width="512", height="512")
    ET.SubElement(assets, "material", name="ground_material", texture="ground_checker",
                  texrepeat="2 2", texuniform="true", reflectance="0")
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", directional="true", dir="0 0 -1")
    ET.SubElement(world, "geom", name="ground", type="plane", size="10 10 0.1",
                  material="ground_material", friction="1 0.005 0.0001",
                  contype="2", conaffinity="1")
    actuators = ET.SubElement(mj, "actuator")
    sensors = ET.SubElement(mj, "sensor")
    qpos_order: list[str] = []

    def add_mesh(name: str, mesh: ET.Element, kind: str) -> str:
        filename = mesh.get("filename")
        prefix = "package://DIAN/"
        if not filename or not filename.startswith(prefix):
            raise ValueError(f"Unsupported mesh path for {name}: {filename}")
        mesh_file = source / filename[len(prefix):]
        if not mesh_file.is_file():
            raise FileNotFoundError(mesh_file)
        asset_name = f"{name}_{kind}_mesh"
        relative_path = f"{kind}/{name}.stl"
        output = copy_mesh(mesh_file, destination / "assets" / relative_path)
        attrs = {"name": asset_name, "file": output.relative_to(destination / "assets").as_posix()}
        if mesh.get("scale"):
            attrs["scale"] = mesh.get("scale")
        ET.SubElement(assets, "mesh", attrs)
        return asset_name

    def add_link(parent: ET.Element, name: str, joint: ET.Element | None = None) -> None:
        body_attrs = {"name": name}
        if joint is None:
            body_attrs["pos"] = "0 0 0.44"
        else:
            body_attrs.update(origin_attributes(joint.find("origin")))
        body = ET.SubElement(parent, "body", body_attrs)
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
            ET.SubElement(body, "site", name="imu", pos="0 0 0", size="0.005", rgba="0 0 0 0")
            ET.SubElement(sensors, "gyro", name="base_gyro", site="imu")
        elif joint.get("type") != "fixed":
            limit = joint.find("limit")
            jname = joint.get("name")
            ET.SubElement(body, "joint", name=jname, type="hinge",
                          axis=joint.find("axis").get("xyz"),
                          range=f'{limit.get("lower")} {limit.get("upper")}',
                          damping="0", armature="0")
            ET.SubElement(actuators, "motor", name=f"{jname}_motor", joint=jname,
                          gear="1", ctrllimited="true", ctrlrange="-25 25")
            qpos_order.append(jname)

        link = links[name]
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        ET.SubElement(body, "inertial", {
            **origin_attributes(inertial.find("origin")),
            "mass": inertial.find("mass").get("value"),
            "fullinertia": " ".join(inertia.get(key) for key in
                                    ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
        })

        visual = link.find("visual")
        if visual is not None:
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Unsupported visual shape on {name}")
            mesh_name = add_mesh(name, mesh, "visual")
            color = visual.find("material/color")
            rgba = color.get("rgba") if color is not None else "0.85 0.9 1 1"
            ET.SubElement(body, "geom", name=name + "_visual", type="mesh", mesh=mesh_name,
                          rgba=rgba, contype="0", conaffinity="0", group="1",
                          **origin_attributes(visual.find("origin")))

        collision = link.find("collision")
        if collision is not None:
            shape = collision.find("geometry")
            box = shape.find("box")
            sphere = shape.find("sphere")
            mesh = shape.find("mesh")
            attrs = {"name": name + "_collision", "contype": "1", "conaffinity": "2",
                     "rgba": "0 0 0 0", "group": "3",
                     **origin_attributes(collision.find("origin"))}
            if box is not None:
                attrs.update(type="box", size=" ".join(str(float(v) / 2) for v in box.get("size").split()))
            elif sphere is not None:
                attrs.update(type="sphere", size=sphere.get("radius"))
            elif mesh is not None:
                attrs.update(type="mesh", mesh=add_mesh(name, mesh, "collision"))
            else:
                raise ValueError(f"Unsupported collision shape on {name}")
            ET.SubElement(body, "geom", attrs)

        for child_joint in children[name]:
            add_link(body, child_joint.find("child").get("link"), child_joint)

    (destination / "assets").mkdir(parents=True, exist_ok=True)
    add_link(world, "base_link")
    home = [0, 0, 0.44, 1, 0, 0, 0] + [DEFAULT_ANGLES[name] for name in qpos_order]
    keyframe = ET.SubElement(mj, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=" ".join(str(x) for x in home))
    ET.indent(mj, space="  ")
    destination.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mj).write(destination / "scene.xml", encoding="utf-8", xml_declaration=True)
    print(f"Wrote {destination / 'scene.xml'} with {len(qpos_order)} joints and {len(links)} links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="DIAN model directory containing urdf/ and meshes_lowpoly/")
    parser.add_argument("--destination", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.source, args.destination)
