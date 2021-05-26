import time 
import torch 
import torch.nn as nn

model.train()

num_epochs = 50
loss_per_iteration = []
iters = []

for epochs in range(1,num_epochs+1):
    tic = time.time()

    loss_per_epoch = 0.0
    batch_num = 0
    for inputs,outputs in train_loader:
        
        torch.cuda.empty_cache()
        inputs,outputs = inputs.to(DEVICE),outputs.to(DEVICE)
        preds = model(inputs)
        loss  = DICEloss(preds,outputs.squeeze(axis=1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_per_epoch += loss
        batch_num +=1

        print("Batch num: {} | Dice Loss:{}".format(batch_num,loss))
    loss_per_iteration.append(loss_per_epoch)
    iters.append(epochs)
    toc = time.time()
  
    print("[{}/{}] Loss : {} Time Taken : {}".format(epochs,num_epochs,loss_per_epoch,(toc - tic)/1000))

    #Saving the model after every epoch
    torch.save(model.state_dict(),'Saved Model ')
    print("Saved the model...")