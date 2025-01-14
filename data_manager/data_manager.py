from collections import defaultdict
from pathlib import Path
import json

from data_manager.datamodel import CollectionRounds, PhotoStream

# TODO
# 1 read PhotoStream metadata (json)
# 2 write PhotoStream metadata (json)
# 3 validate / sync PhotoStream metadata (json)


""" 
For a set of images we want to keep track what the order of the image is
and what the weather/lighting condition were at the time of taking the image.
 
"""

# collection = CollectionRounds()

# def collection_round_()


def collection_round_to_streams(
    collection_rounds: CollectionRounds,
) -> list[PhotoStream]:
    photo_info_groups = defaultdict(list)
    photo_streams = []
    for collection_round in collection_rounds:
        for photo_info in collection_round:
            photo_info_groups[photo_info.photostream_id].append(photo_info)
        photo_streams.append(
            PhotoStream(
                photo_info.photostream_id,
                photo_info_groups[photo_info.photostream_id],
                collection_round_tags=collection_round.tags,
                asset=collection_round.asset,
            )
        )
    return photo_streams
    # photo_info_groups[photo_info.photostream_id]
    # for img_info in img_infos:
    # return groups


def save_collection(collection: CollectionRounds, output_file_path: Path = None):
    if not output_file_path:
        output_file_path = collection.dir_path.parent
    with open(output_file_path, "w+") as output_file:
        json.dump(collection, output_file)
