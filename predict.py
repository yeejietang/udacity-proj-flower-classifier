import argparse 
import util 
import json 
import numpy as np 
import torch 
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

def get_predict_inputs():
    parser = argparse.ArgumentParser(description='Inputs for model predicting')
    parser.add_argument('image_file', type=str, help='Required image file to predict for')
    parser.add_argument('checkpoint', type=str, help='Trained model checkpoint to load from')
    parser.add_argument('--top_k', type=int, default=5, help='Top K results to print')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='JSON file to provide mapping of categories to real names')
    parser.add_argument('--gpu', default=False, action='store_true', help='Flag to use GPU true')
    
    # Prints the namespace
    print(parser.parse_args())
    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Define the transform
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    
    # Opne the image
    pil_image = Image.open(image)
    
    # Transforms the image
    pil_image = img_transform(pil_image).float()
    
    # Put in numpy array
    np_image = np.array(pil_image)
    
    # Define means & standard deviations
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    # Normalize
    np_image = (np.transpose(np_image, (1,2,0)) - mean)/std
    np_image = np.transpose(np_image, (2,0,1))
    
    return np_image


def predict_image(image_path, model, class_to_idx, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model & its class to index mapping
    Returns: top 5 probabilities
    top 5 classes
    '''
    
    # Implement the code to predict the class from an image file
    image_arr = process_image(image_path)
    image_tensor = torch.FloatTensor(image_arr)
    image = Variable(image_tensor)
    image = image.unsqueeze(0).float()
    
    # Use GPU if possible!
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        
    # Forward pass to get predictions
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_ps, top_idx = ps.topk(topk)
        
    # Create invert map for class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Return probabilities & classes
    probs = top_ps.data[0].tolist()
    classes = [idx_to_class[idx] for idx in top_idx.data[0].tolist()]
    
    return probs, classes


def main():
    
    # First, get input arguments
    args = get_predict_inputs()
    
    # Load checkpoint of previously trained model
    model, class_to_idx = util.load_saved_model(args.checkpoint, args.gpu)
    
    # Predict for image file
    sample_prob, sample_classes = predict_image(args.image_file, model, class_to_idx, args.top_k, args.gpu)
    
    # Label mapping file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Print statements of predictions
    print(f"\nThis image most likely contains a {cat_to_name[sample_classes[0]]} flower with a probability of {round(sample_prob[0]*100)}%")
    
    # Print for top_k number of classes
    print(f"\nThese are the top {args.top_k} predicted class(es) and their respective probabilities:")
    
    for i in range(args.top_k):
        flower_name = cat_to_name[sample_classes[i]] 
        prob = round(sample_prob[i]*100)
        print(f"{flower_name}: {prob}%")
    
    return 


if __name__ == "__main__":
    main()
