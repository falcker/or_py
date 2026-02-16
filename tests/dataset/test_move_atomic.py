def test_atomic_move_filesystem(dataset):
    img = dataset.train_COCO.images[0]

    # create real file
    (dataset.train_root_path / img.file_name).write_text("x")

    moved = dataset.move_images_atomic(
        source="train",
        target="valid",
        asset_names={"F01"},
    )

    assert moved == 2  # both F01 images
    assert not (dataset.train_root_path / img.file_name).exists()
    assert (dataset.valid_root_path / img.file_name).exists()
