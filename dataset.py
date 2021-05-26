from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import torchvision.transforms.functional as TF

color_scheme = pd.read_csv('class_dict_seg.csv')

color_scheme

class DroneDataset(Dataset):
    def __init__(self,images_path,labels_path):
        self.images = datasets.ImageFolder(images_path,
                                       transform=transforms.Compose([
                                          transforms.Resize((256,256)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                       ]))
        self.labels = datasets.ImageFolder(labels_path,transform=transforms.Compose([
                                          transforms.Grayscale(),
                                          transforms.Resize((256,256)),
                                          transforms.ToTensor()
        ]))

    def __getitem__(self,index): 
        img_output = self.labels[index][0]
        img_output = 255*img_output
        img_output = img_output.to(torch.int64)
        return self.images[index][0],img_output
  
    def __len__(self):
        return len(self.images)

dataset = DroneDataset(images,labels)

dataset[0][1].shape

#Split the data into train and val dataset
n_val = 10

train_dataset,val_dataset = random_split(dataset,[len(dataset)-n_val,n_val],generator=torch.Generator().manual_seed(42))

#Make the data_loader now so that the data is ready for training
batch_size = 4

train_loader = DataLoader(train_dataset,batch_size)
test_loader  = DataLoader(val_dataset,batch_size*2)
