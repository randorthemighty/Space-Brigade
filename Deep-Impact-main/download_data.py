import os
import gdown
import zipfile

ID = "1v48d_JOy0RZFBu6AoAyr35XhGiNQaN68"
OUT = os.sep.join([".", "resources", "geodata.zip"])

print(OUT)

gdown.download(id=ID, output=OUT)

with zipfile.ZipFile(OUT, "r") as zip_ref:
    zip_ref.extractall(os.sep.join([".", "resources"]))
