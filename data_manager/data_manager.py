from collections import defaultdict
from pathlib import Path
import json

from datamodel import CollectionRounds

# TODO
# 1 read PhotoStream metadata (json)
# 2 write PhotoStream metadata (json)
# 3 validate / sync PhotoStream metadata (json)


""" 
For a set of images we want to keep track what the order of the image is
and what the weather/lighting condition were at the time of taking the image.
 
"""

# collection = CollectionRounds()


def sort_collection_rounds_by_streams(collection_rounds: CollectionRounds):
    for collection_round in collection_rounds:
        collection_round
    #     groups = defaultdict(list)
    # for img_info in img_infos:
    #     groups[img_info.filename.guid].append(img_info)
    # return groups


def save_collection(collection: CollectionRounds, output_file_path: Path = None):
    if not output_file_path:
        output_file_path = collection.dir_path.parent
    with open(output_file_path, "w+") as output_file:
        json.dump(collection, output_file)
