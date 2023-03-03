valid_label_type_dict = {
    "name": str,
    "image": str,
    "state": bool,
    "map": str,
    "vehicle":
        {
            "location": {
                "x": float,
                "y": float,
                "z": float,
            },
            "rotation": {
                "pitch": float,
                "yaw": float,
                "roll": float,
            }
        },
    "camera":
        {
            "location": {
                "x": float,
                "y": float,
                "z": float,
            },
            "rotation": {
                "pitch": float,
                "yaw": float,
                "roll": float,
            },
            "fov": int
        }
}
