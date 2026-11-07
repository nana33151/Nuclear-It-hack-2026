"""Shared camera aliases for geometry; original detection IDs remain unchanged."""
def normalize_camera_id(camera_id: str) -> str:
    for prefix in ("camera_new_", "cam_"):
        if camera_id.startswith(prefix) and camera_id[len(prefix):].isdigit():
            return "cam" + str(int(camera_id[len(prefix):]))
    return camera_id
