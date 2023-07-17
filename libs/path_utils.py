import os
import random


def get_filelist_from_dir(path, recursive=False, filetypes=["jpg", "jpeg", "png"], max_count=False):
    '''create a list of all images in the directory. if the path is a file, return a list with only the file.'''

    if os.path.isdir(path):
        # create index of all images to predict
        filelist = []
        for (root,dirs,files) in os.walk(path, topdown=True):
                for i, file in enumerate(files):
                    if file.split(".")[-1] in filetypes:
                        img_path = os.path.join(root,file)
                        filelist.append(img_path)

                if not recursive:
                    break

        # sample random images only if there are more than max_count
        if max_count >= 1:
            if len(filelist) > max_count:
                filelist = random.sample(filelist, max_count)
    
    # error handling
    elif os.path.isfile(path):
        print(f"[WARNING] {path} should be a directory. Proceding anyway.")
        filelist = path

    return filelist


def get_filename_from_path(path):
    filename = "image" if path is None else os.path.splitext(os.path.basename(path))[0]
    return filename


def add_name_attribute(imagepath, target_dir="images/previews", attribute="shape"):
    '''changes output directory and adds attribute to filename'''

    # TODO: make output directory change optional
    basename = os.path.basename(imagepath)
    namesplit = os.path.splitext(basename)
    savename = namesplit[0] + "_" + attribute + namesplit[1]
    savepath = os.path.join(target_dir, savename)
    return savepath


def modify_path(path, attrib=None, out_dir= None, ext=None):
    root, file = os.path.split(path)
    name, extension = os.path.splitext(file)

    if out_dir is not None:
        root = out_dir

    if ext is not None:
        extension = ext
        
    if attrib is not None:
        attrib = "_" + attrib

    result_path = os.path.join(root, name + attrib + "." + extension)
    return result_path
