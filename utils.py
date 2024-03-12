import os
from PIL import Image
import pickle
import tensorflow as tf

model = pickle.load(open("model/model_efficientnet_v1.pkl",'rb'))


def f_resize_raw_image(src_dir, tgt_dir, target_size):
    
    ls_src_dir_files = [ os.path.join(src_dir, i) for i in os.listdir(src_dir)]
    try:
        ls_src_dir_files.remove(src_dir+'/.DS_Store')
    except:
        print('Not removed')

    for i, img_path in enumerate(ls_src_dir_files):
        print(img_path)
        print(f'[+] Processing file {img_path}...')
        image = Image.open(img_path)
        image = image.convert('RGB') #convert png to jpg
        a = image.resize(target_size)
        a.save(tgt_dir+'/'+str(i)+'.jpg')


def f_predict(image_generator):
    pred_hist = model.predict(image_generator)
    print(pred_hist)
    if pred_hist[0][0]<0.5:
        return False
    else:
        return True


def f_clear_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate over each file and delete it
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        # Check if it's a file (not a directory) before attempting to delete
        if os.path.isfile(file_path):
            os.remove(file_path)
            #print(f"Deleted: {file_path}")


def f_remove_files_in_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file and delete it
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        # Check if it's a file (not a directory) before attempting to delete
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")