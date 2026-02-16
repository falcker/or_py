def test_time_aware_split(dataset):
    result = dataset.rebalance_time_aware(
        train_ratio=0.5,
        valid_ratio=0.25,
        test_ratio=0.25,
    )

    assert sum(result.values()) == 3

    # F01 images should split chronologically
    train_dates = [
        img.date_captured
        for img in dataset.train_COCO.images
        if img.asset_name == "F01"
    ]

    assert train_dates == sorted(train_dates)


def test_no_asset_leakage_after_rebalance(dataset):
    dataset.rebalance_time_aware()

    report = dataset.full_audit_report()

    assert report["leakage"]["asset_leakage"] == {}
