from PIL import Image
import requests
import os

def read_img_from_url(url):
  return Image.open(requests.get(url, stream=True).raw)

def create_folder(dir_name,
                  root_dir):
  new_path = os.path.join(root_dir, dir_name)

  if dir_name not in os.listdir(root_dir): # check if already exists
    os.mkdir(new_path)