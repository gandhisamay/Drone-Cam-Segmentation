images = 'dataset/semantic_drone_dataset/Original_Images'
rgb_masks = 'RGB_color_image_masks'
labels = 'dataset/semantic_drone_dataset/Labels'

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
        #Manipulate the label Images
        mask = np.array([[0,0],[1,1],[2,0],[3,3],[4,1],[5,5],[6,0],[7,5],[8,3],[9,2],[10,0],[11,2],[12,2],[13,0],[14,0],[15,5],[16,5],[17,6],[18,6],[19,3],
                          [20,3],[21,0],[22,0],[23,0]])
        for i in range(0,24):
          img_output[img_output == i] = mask[i][1]

        img_output = img_output.to(torch.int64)
        return self.images[index][0],img_output

    def __len__(self):
        return len(self.images)

dataset = DroneDataset(images,labels)

n_val = 10

train_dataset,val_dataset = random_split(dataset,[len(dataset)-n_val,n_val],generator=torch.Generator().manual_seed(42))

#Make the data_loader now so that the data is ready for training
batch_size = 4

train_loader = DataLoader(train_dataset,batch_size)
test_loader  = DataLoader(val_dataset,batch_size*2)
