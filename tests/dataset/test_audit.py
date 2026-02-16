def test_audit_integrity(dataset):
    report = dataset.full_audit_report()

    assert report["integrity"]["duplicate_image_ids"] == []
    assert report["leakage"]["asset_leakage"] == {}

