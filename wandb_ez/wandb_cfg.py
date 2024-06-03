__all__ = ["wandb_cfg"]

wandb_cfg = {
    "key": "463b69d4a0864d56d8a99e38d67a896af404b7cc",
    "project": "wandb_plugin_dev_test",
    "sweep": True,
    "sweep_param": ["bit_width_list"],
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 5,
    "sweep_id": None
}