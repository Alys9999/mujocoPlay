from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1_SCENES = (
    REPO_ROOT / "phase1" / "assets" / "phase1_franka_scene.xml",
    REPO_ROOT / "phase1" / "assets" / "phase1_franka_scene_small_box.xml",
    REPO_ROOT / "phase1" / "assets" / "phase1_franka_scene_cylinder.xml",
)
PANDA_XML = REPO_ROOT / "third_party" / "franka_emika_panda" / "panda.xml"


def _parse_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _parse_floats(values: str) -> list[float]:
    return [float(value) for value in values.split()]


@pytest.mark.fast
def test_phase1_scenes_source_wrist_camera_from_third_party_panda_xml():
    for scene_path in PHASE1_SCENES:
        root = _parse_xml(scene_path)
        include_files = [node.attrib["file"] for node in root.findall("include")]
        assert "../../third_party/franka_emika_panda/panda.xml" in include_files

        camera_names = [node.attrib.get("name") for node in root.findall(".//camera")]
        assert "arm_attached" not in camera_names


@pytest.mark.fast
def test_third_party_panda_xml_defines_hand_wrist_camera_pose():
    root = _parse_xml(PANDA_XML)

    hand_body = root.find(".//body[@name='hand']")
    assert hand_body is not None

    camera = hand_body.find("camera[@name='arm_attached']")
    assert camera is not None
    assert _parse_floats(camera.attrib["pos"]) == pytest.approx([0.0, -0.055, 0.08])
    assert _parse_floats(camera.attrib["xyaxes"]) == pytest.approx([1.0, 0.0, 0.0, 0.0, -0.385806, 0.92258])
    assert float(camera.attrib["fovy"]) == pytest.approx(70.0)
