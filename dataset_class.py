import os 
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision import transforms 

class SketchDataset(Dataset):
    def __init__(self,image_path,label_path):
        self.image_path = image_path 
        self.label_path = label_path
        
        #Define the transforms for input images
        transform_input = transforms.Compose([
            transforms.Resize((240,360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        #Input images loading
        img_dataset = datasets.ImageFolder(self.image_path,transform = transform_input)
        self.img_dataset = img_dataset
        
    def __getitem__(self,index):
        #Sketch loading
        for img in os.listdir(self.label_path):
            if img.endswith(str(index) + '_05.png'):
                sketch = Image.open(self.label_path + './' + img)
        #transforms for sketches
        sketch = TF.to_tensor(
                    TF.resize(
                        sketch,size=(240,360)
                    )
                 )
        return self.img_dataset[index][0],sketch 
    
    def __len__(self):
        return len(self.img_dataset)
    
    dataset = SketchDataset(imgs,sketches)