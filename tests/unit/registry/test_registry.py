import pytest

from benchmark.core.registry.registry import ComponentRegistry, create_default_registry


@pytest.mark.fast
def test_component_registry_rejects_duplicate_entries():
    registry = ComponentRegistry()
    registry.register("example", lambda: 1)
    with pytest.raises(KeyError):
        registry.register("example", lambda: 2)


@pytest.mark.fast
def test_default_registry_contains_core_components():
    registry = create_default_registry()
    assert "normal_pick_place" in registry.keys("presets")
    assert "franka_panda_2f_v1" in registry.keys("robot_profiles")
    assert "pick_place" in registry.keys("task_definitions")
    assert "hidden_physics_blocks_v1" in registry.keys("object_sets")
    assert "random" in registry.keys("policy_adapters")
    assert "jsonl" in registry.keys("trace_sinks")
