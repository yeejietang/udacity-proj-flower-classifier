import torch 
import os
from torchvision import datasets, transforms, models 

def preprocess_data(data_dir, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    '''
    Inputs: 
    data_dir = data directory (where we will get our train, validation, test data)
    norm_mean = mean of images, to normalize (optional)
    norm_std = std of images, to normalize (optional)
    
    Outputs:
    trainloader, valloader, testloader = data loaders for train, val, test sets
    train_class_to_idx = class to index mapping from training set
    class size = number of classes in training set
    '''
    # Define directories for train, val, test
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(p=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    # Get the class_to_idx mapping
    train_class_to_idx = train_dataset.class_to_idx
    
    # Get class size
    output_size = len(train_dataset.classes)
    
    return trainloader, valloader, testloader, train_class_to_idx, output_size
    
    
def save_trained_model(model, input_size, epochs, output_size, arch, learning_rate, class_to_idx, optimizer, criterion, save_dir):
    '''
    Inputs: many things related to state_dict that we want to save
    Outputs: None. Prints message with save directory
    '''
    saved_state_dict = {
        'input_size': input_size, 
        'epochs': epochs, 
        'output_size': output_size, 
        'arch': arch, 
        'hidden_layers': [each.out_features for each in model.classifier if hasattr(each, 'out_features') == True], 
        'learning_rate': learning_rate, 
        'state_dict': model.state_dict(), 
        'class_to_idx': class_to_idx, 
        'optimizer_dict': optimizer.state_dict(), 
        'criterion_dict': criterion.state_dict(), 
        'classifier': model.classifier 
    }
    
    # When save_dir is defined
    if not save_dir is None:
        
        # Create directory if it doesn't already exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Define save path    
        save_path = save_dir + '/checkpoint.pth' 
        
    # When save_dir is None
    else:
        save_path = 'checkpoint.pth'
    
    torch.save(saved_state_dict, save_path) 
    print("Model checkpoint saved at {}".format(save_path)) 
    
    return 
    
    
def load_saved_model(checkpoint):
    '''
    Inputs: 
    Model checkpoint
    Outputs: 
    Model - rebuilt model from saved checkpoint
    Class to index mapping
    '''
    # Load the model, find out what architecture was saved into the checkpoint
    trained_model = torch.load(checkpoint)
    arch = trained_model['arch'] 
    
    # Rebuilds according to architecture
    if arch == 'vgg':
        loaded_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        loaded_model = models.alexnet(pretrained=True)
    else:
        print("Only 'vgg' and 'alexnet' are currently supported.")
        sys.exit() 
    
    # Freeze params
    for param in loaded_model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier, state_dict, get class_to_idx mapping
    loaded_model.classifier = trained_model['classifier']
    loaded_model.load_state_dict(trained_model['state_dict'])
    class_to_idx = trained_model['class_to_idx']
    
    return loaded_model, class_to_idx
    