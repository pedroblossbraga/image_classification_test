from load_data import load_data
from utils import create_folder, read_img_from_url
from load_data import load_data
import os

def save_images(images_urls, 
                output_dir):
    """ 
    Given URLs, read the image and save (it does not exist already)
    """
    N_imgs = len(images_urls)
    for i, url in enumerate(images_urls):
        img = read_img_from_url(url)
        
        new_img_name = 'img_{}.jpg'.format(i)
        
        print('({}/{}) ({:.3f}%) ({}) - {}'.format(
            i+1, N_imgs, (i+1)*100/N_imgs,
            new_img_name,
            output_dir
        ))
        if new_img_name not in os.listdir(output_dir):
            img.save(os.path.join(output_dir, new_img_name))

def setup_dir(
    root_path
):
    # create a folder to store images for each class
    
    create_folder(dir_name = 'image_dataset',
                  root_dir = root_path)
    
    classes = ['not st george', 'st george']
    for class_name in classes:
        create_folder(dir_name = class_name,
                        root_dir = os.path.join(root_path, 'image_dataset'))
        
    # read datasets of URLs
    df_nongeorges, df_georges = load_data(path = os.path.join(root_path, 'test_assignment_cv'))
    
    save_images(images_urls = df_nongeorges[df_nongeorges.columns[0]].values, 
                output_dir = os.path.join(root_path, 'image_dataset', 'not st george'))
    
    save_images(images_urls = df_georges[df_georges.columns[0]].values, 
                output_dir = os.path.join(root_path, 'image_dataset', 'st george'))