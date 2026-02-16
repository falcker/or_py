def test_index_build(dataset):
    ds = dataset.train_COCO
    assert ds.get_image(1).id == 1
    assert len(ds.get_annotations(1)) == 1
