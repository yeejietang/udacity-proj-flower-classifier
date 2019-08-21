import torch
import os
import argparse 
import util 
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.autograd import Variable

def get_train_inputs():
    parser = argparse.ArgumentParser(description='Inputs for model training')
    parser.add_argument('data_dir', type=str, help='Required data directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg', help='Model Architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of units in hidden layers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpu', default=False, action='store_true', help='Flag to use GPU true')
    
    # Prints the namespace
    print(parser.parse_args())
    return parser.parse_args()


def load_pretrained_model(arch='vgg'):
    '''
    Inputs: 
    arch = can be 'vgg' or 'alexnet' (default: vgg)
    
    Outputs:
    pretrained model
    input size of model
    '''
    if arch == 'vgg':
        load_pre_model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        load_pre_model = models.alexnet(pretrained=True)
        input_size = 9216
        
    # Freeze training for all layers
    for param in load_pre_model.parameters():
        param.require_grad = False
        
    return load_pre_model, input_size


def fit_new_classifier(model, input_size, hidden_units, output_size, drop_prob = 0.5):
    '''
    Inputs:
    model = pretrained model to be fit with new classifier
    input_size = input size of pretrained model
    hidden_units = number of hidden units
    output_size = number of classes to train for
    drop_prob = dropout probability (default = 0.5)
    
    Outputs:
    None
    '''
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=drop_prob)),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=drop_prob)),
        ('fc3', nn.Linear(hidden_units, hidden_units)),
        ('relu3', nn.ReLU()),
        ('drop3', nn.Dropout(p=drop_prob)),
        ('fc4', nn.Linear(hidden_units, output_size)), 
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Fit new classifier
    model.classifier = classifier
    return 


def train_model(model, epochs, trainloader, valloader, optimizer, criterion, gpu):
    '''
    Inputs: 
    model = model to train
    epochs = number of epochs to train
    trainloader, valloader = training and validation data loaders
    optimizer = optimizer to user
    criterion = loss function
    gpu = flag for GPU availability

    Outputs:
    trained model
    '''
    # Train on GPU if available!
    if gpu and torch.cuda.is_available():
        model.cuda()

    # Define variables to loop through
    steps = 0
    print_every = 20
    running_loss = 0
    
    # Loop for the number of epochs
    for epoch in range(epochs):
    
        model.train()

        # Loop through trainloader
        for inputs, labels in iter(trainloader):
            steps += 1
            inputs, labels = Variable(inputs), Variable(labels)
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track loss and accuracy on validation set
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                for inputs, labels in iter(valloader):
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    with torch.no_grad():
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels).data
                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(valloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valloader):.3f}")

                running_loss = 0
                model.train()
                
    print("Model has completed training") 
    return model        


def test_model(model, testloader, criterion, gpu=True):
    '''
    Inputs: 
    model = trained model to test
    testloader = test data loader
    criterion = loss function
    gpu = flag for GPU availability

    Outputs:
    None
    '''
    # Train on GPU if available!
    if gpu and torch.cuda.is_available():
        model.cuda()
        
    model.eval()

    test_accuracy = 0
    test_loss = 0

    # forward pass
    for inputs, labels in iter(testloader):
        inputs, labels = Variable(inputs), Variable(labels)
        if gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels).data
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f} "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}") 
    return 


def main():
    
    # First, get input arguments
    args = get_train_inputs()
    
    # Get data directory (mandatory input)
    data_dir = args.data_dir
    
    # Get data loaders, after preprocessing
    trainloader, valloader, testloader, class_to_idx, output_size = util.preprocess_data(data_dir)
    
    # Load a pretrained model
    model, input_size = load_pretrained_model(arch = args.arch)
    
    # Fit a new classifier to loaded model
    fit_new_classifier(model, input_size, args.hidden_units, output_size, drop_prob = 0.5)
    
    # Prints architecture of new model
    print(model)
    
    # Define loss function & optimizer
    learning_rate = args.learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train model
    model = train_model(model, args.epochs, trainloader, valloader, optimizer, criterion, args.gpu)
    
    # Do validation on the test set
    test_model(model, testloader, criterion)
    
    # Save model
    util.save_trained_model(model, input_size, args.epochs, output_size, args.arch, learning_rate, 
                            class_to_idx, optimizer, criterion, args.save_dir)
    
    return 


if __name__ == "__main__":
    main()