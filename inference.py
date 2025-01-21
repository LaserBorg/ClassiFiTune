import torch
import os
import cv2

from libs.dataset_utils import load_imagefile_to_tensor, get_transforms, convert_image_to_cv, get_model_data
from libs.path_utils import get_filelist_from_dir


def predict(model, img, apply_softmax=False):
    with torch.no_grad():
        class_outputs = model(img)

    # normalize outputs to range[0-1] and sum=1
    if apply_softmax:
        class_outputs = torch.softmax(class_outputs, dim=1)

    top_probability, top_prediction = torch.max(class_outputs, 1)

    # [0] because top_probability and top_prediction are tensors with one element
    return top_prediction[0], top_probability[0]


def run_inference(checkpoint_path, images_path, force_CPU=True, apply_softmax=False, single_output=False, output_level=0):

    # infer all images
    if output_level < 2:
        max_count = False
    
    # if output_level == 2, infer an show {max_count} images
    if output_level == 2:
        max_count = 20
    
    # if output_level == 3, infer and show all image but stop at each one
    elif output_level > 2:
        max_count = False

    
    # create list of images to predict
    if os.path.isdir(images_path):
        filelist = get_filelist_from_dir(images_path, max_count=max_count)
    else:
        filelist = [images_path]
        single_output = True
        

    input_size, class_names = get_model_data(checkpoint_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not force_CPU else "cpu")
    if output_level >= 3:
        print(f"[INFO] running inference on {device}")
    
    model = torch.load(checkpoint_path)
    model = model.to(torch.device(device))

    # set dropout and batch normalization layers to evaluation mode before running inference
    model.eval()

    inference_transform = get_transforms(input_size)["val"]

    model = model.to(device)
    model.eval()    

    # initialize lists of resulting top-predictions and their probabilities
    top_probabilities_list = []
    top_predictions_list = []

    for img_path in filelist:
        img = load_imagefile_to_tensor(img_path, transform=inference_transform, device=device)
        top_prediction, top_probability = predict(model, img, apply_softmax=apply_softmax)
        
        # get top prediction and its probability
        top_predicted_label = class_names[top_prediction]
        top_predictions_list.append(top_predicted_label)
        top_probabilities_list.append(top_probability)
        

        # print top prediction
        if output_level == 1:
            print(f"predicted:\t{top_predicted_label}\t({top_probability * 100:.2f} %)")

        # show all images
        elif output_level == 2:
            img = convert_image_to_cv(img)
            cv2.imshow(top_predicted_label, img)
            cv2.waitKey(1)
        
        # stop at each image
        elif output_level ==3:
            cv2.waitKey(0)
        
    if single_output:
        return top_predictions_list[0], top_probabilities_list[0]
    else:
        return filelist, top_predictions_list, top_probabilities_list


if __name__ == "__main__":
    
    modelfile = "mobilenet_v3_large.pt"
    checkpoint_path = os.path.join("checkpoints/", modelfile)

    # print("\nINFERENCE ON SINGLE IMAGE")
    # img_path = "images/previews/20230619-101439_(144.7_2.47_-0.67)_rectified_hinges.jpg"
    # top_prediction, top_probability = run_inference(checkpoint_path, 
    #                                                 img_path, 
    #                                                 force_CPU=True,
    #                                                 apply_softmax=False,
    #                                                 output_level=2)
    # print(f"{img_path} :\t{top_prediction}\t({top_probability * 100:.2f} %)")


    print("\nINFERENCE OF FOLDER (GPU)")
    images_dir = './dataset'
    filelist, top_predictions_list, top_probabilities_list = run_inference(checkpoint_path, 
                                                                           images_dir, 
                                                                           force_CPU=False,
                                                                           apply_softmax=False,
                                                                           output_level = 2)
    for file in filelist:
        prediction = top_predictions_list[filelist.index(file)]
        probability = top_probabilities_list[filelist.index(file)]
        print(f"{file} :\t{prediction}\t({probability * 100:.2f} %)")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
