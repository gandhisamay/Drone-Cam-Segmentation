#Suppose you have an Image

img = Image.open(location of Image)

img_preprocessed = TF.normalize(
    TF.to_tensor(
        TF.resize(
            img,size=(256,256)
            )
        )
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

output = model(img_preprocessed)

output = torch.argmax(F.softmax(output,dim=1),axis=1)

#The final output is obtained
