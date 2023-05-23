
from torchvision import models
import torch.nn as nn


"""
This helper function sets the ``.requires_grad`` attribute of the parameters in the model 
to False when we are feature extracting. By default, when we load a pretrained model all 
of the parameters have ``.requires_grad=True``, which is fine if we are training from scratch
or finetuning. However, if we are feature extracting and only want to compute gradients for 
the newly initialized layer then we want all of the other parameters to not require gradients. 
This will make more sense later.
"""
def set_parameter_requires_grad(model, train_deep):
    if not train_deep:
        for param in model.parameters():
            param.requires_grad = False



"""
Model Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def initialize_model(model_name, num_classes, train_deep):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    
    if model_name == "resnet18":
        model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "vgg11_bn":
        model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224


    elif model_name == "densenet121":
        model_ft = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224


# NEW ------------------------------------------

    elif model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "mobilenet_v3_large":
        model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "regnet_y_16gf":
        model_ft = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "efficientnet_v2_s":
        model_ft = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "efficientnet_v2_m":
        model_ft = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    

    elif model_name == "convnext_base":
        model_ft = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "swin_v2_b":
        model_ft = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes) 
        input_size = 224





    # Inception v3
    # ~~~~~~~~~~~~
    # https://arxiv.org/pdf/1512.00567v1.pdf
    # This network is unique because it expects (299,299) sized images and has two output layers when training. 
    # The second output is known as an auxiliary output and is contained in the AuxLogits part of the network. 
    # The primary output is a linear layer at the end of the network. Note, when testing we only consider the 
    # primary output. The auxiliary output and primary output of the loaded model are printed as:

    #     (AuxLogits): InceptionAux(
    #         ...
    #         (fc): Linear(in_features=768, out_features=1000, bias=True)
    #         )
    #         ...
    #     (fc): Linear(in_features=2048, out_features=1000, bias=True)

    # To finetune this model we must reshape both layers. This is accomplished
    # with the following

    #     model.AuxLogits.fc = nn.Linear(768, num_classes)
    #     model.fc = nn.Linear(2048, num_classes)


    elif model_name == "inception_v3":
        model_ft = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, train_deep)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size
