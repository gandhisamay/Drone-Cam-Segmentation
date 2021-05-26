import torch 
import torch.nn.functional as F 

def DICEloss(preds,outputs,smooth=1):
  preds = F.softmax(preds,dim=1)
  labels_one_hot = F.one_hot(outputs, num_classes = 23).permute(0,3,1,2).contiguous()
  intersection = torch.sum(preds*labels_one_hot)
  total = torch.sum(preds*preds) + torch.sum(labels_one_hot*labels_one_hot)
  return 1-((2*intersection + smooth)/(total))