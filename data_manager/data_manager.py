from pathlib import Path
import json

from datamodel import CollectionRoundSet

# TODO
# 1 read PhotoStream metadata (json)
# 2 write PhotoStream metadata (json)
# 3 validate / sync PhotoStream metadata (json)


""" 
For a set of images we want to keep track what the order of the image is
and what the weather/lighting condition were at the time of taking the image.
 
"""

collection = CollectionRoundSet()


def save_collection(collection: CollectionRoundSet, output_file_path: Path = None):
    if not output_file_path:
        output_file_path = collection.dir_path.parent
    with open(output_file_path, "w+") as output_file:
        json.dump(collection, output_file)
