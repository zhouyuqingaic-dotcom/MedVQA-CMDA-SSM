from config.stage1_train_config_mimic_cxr import TrainConfig
from datas.mimic_cxr_datasets import MIMICCXRDataset

cfg = TrainConfig()

dataset = MIMICCXRDataset(
    csv_path=cfg.mimic_cxr_metadata_csv,
    image_root=cfg.mimic_cxr_image_root,
    report_root=cfg.mimic_cxr_report_root,
    target_section=cfg.mimic_cxr_target_section,
    load_report=cfg.mimic_cxr_load_report,
    allowed_view_positions=cfg.mimic_cxr_view_positions,
    drop_empty_target=cfg.mimic_cxr_drop_empty_target,
    cache_dir=cfg.mimic_cxr_cache_dir,
    use_indices_cache=True,
    rebuild_indices_cache=True,
    cache_prefix=cfg.mimic_cxr_cache_prefix,
)

print(f"新缓存构建完成，有效样本数：{len(dataset)}")