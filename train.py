import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from ViT import vit
from utils import WarmupLinearSchedule, getdata

dataset = 'cifar100'
def trainepoch(model, trainloader, criterion, opt, device) :
    '''
    Train 1 epoch
    Input:
    model - network to be trained
    trainloader - training data
    criterion - loss function
    opt - optimizer
    device - hardware on which training is performed
    Output:
    model - one epoch trained model
    loss - per epoch loss function
    '''
    epochloss = 0
    acc = 0
    total = 0
    for b, (x,y) in enumerate(trainloader) :
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        #forward pass
        logits = model(x)
        predicted = torch.argmax(logits, dim=-1)
        acc += torch.eq(predicted, y).sum()
        total += y.shape[0]
        #backward pass
        loss = criterion(logits, y)
        loss.backward()
        clip = 5
        torch.nn.utils.clip_grad_norm(model.parameters(),clip)
        opt.step()

        #accumulated loss
        print("Batch ", b, "--------- loss = ", loss.item())
        epochloss += loss

    return model, epochloss/b, acc/total

def train(model, trainloader, device, epochs=5, loss='CE', optimizer='SGD', lr=0.1) :
    '''
    Train a model
    Input:
    model - network to be trained
    trainloader - dataloader for training
    device - hardware on which to perform training
    epochs - number of training epochs
    loss - type of loss function used
    optimizer - type of optimizer used
    lr - learning rate
    Output:
    model - trained model
    finalloss - loss at the end of training
    '''
    model.train()
    #define loss function
    if loss == 'CE' :
        criterion = nn.CrossEntropyLoss()
    else :
        print("Undefined loss function")
        return None, None

    #define optimizer
    if optimizer == 'SGD' :
        opt = optim.SGD(params=model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=5e-4)
    else :
        print("Undefined optimizer")
        return None, None

    #learning rate scheduler
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=opt,
    #                                                  milestones=[epochs//2, int(3*epochs//4)],
    #                                                  gamma=0.1)
    scheduler = WarmupLinearSchedule(opt, warmup_steps=1, t_total=epochs)

    loss = []
    accumulate = 0.0
    acc = []
    for i in range(epochs) :
        model, epochloss, accuracy = trainepoch(model, trainloader, criterion, opt, device)
        #print(scheduler.get_last_lr())    #print to check learning rate value every epoch
        scheduler.step()
        print("Epoch ", i, "--------- avg. loss: %.3f" %(epochloss.item()))
        loss.append(epochloss.item())
        accumulate += accuracy.item()
        if (i+1) % 5 == 0:
          acc.append(accumulate/5.0)
          accumulate = 0.0
    torch.save(model.state_dict(), f"/u/yuli9/cs533_final_project/models/{dataset}/final.pt")
    return model, epochloss, loss, acc

def test(model, testloader, device, loadpath=None) :
    '''
    Evaluates the inference accuracy of the model

    Input:
    model - network to be evaluated
    testloader - dataloader for testing
    device - hardware on which to evaluate

    Output:
    acc - accuracy of the model
    '''
    if loadpath is not None :
      ckpt = torch.load(loadpath)    #load checkpoint
      model.load_state_dict(ckpt)    #load trained model

    model.eval()
    acc = 0
    total = 0

    with torch.no_grad() :
        for b, (x, y) in enumerate(testloader) :
            x, y = x.to(device), y.to(device)
            #obtain predicted class
            predicted = model(x)
            predicted = torch.argmax(predicted, dim=-1)

            #calculate the accuracy
            acc += torch.eq(predicted, y).sum()
            total += y.shape[0]

    return acc.item()/total

def main():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Beginning training on ", device)

    nclasses = 100 if dataset == 'cifar100' else 10
    ipch = 3
    image_size = 224
    num_layers = 32
    d_model = 1280
    mlp_ratio = 5120/1280
    nheads = 16
    patch_size = 32
    num_epoch=1

    trainloader, testloader = getdata(dataset, image_size)
    print("Training: no. of batches = ", len(trainloader), "no. of samples = ", len(trainloader.dataset))
    print("Inference: no. of batches = ", len(testloader), "no. of samples = ", len(testloader.dataset))

    model = vit(ipch=ipch, Nclasses=nclasses, image_size=image_size, num_layers=num_layers, patch_size=patch_size, d_model=d_model, nhead=nheads, mlp_ratio = mlp_ratio).to(device)
    model, finalloss, loss, accuracy = train(model, trainloader, device, epochs=num_epoch, lr=2e-4)

    print("\nFinal avg. training loss: %.3f" %(finalloss.item()))
    acc = test(model, testloader, device, loadpath=f"/u/yuli9/cs533_final_project/models/{dataset}/final.pt")
    print("Final model accuracy = ", acc*100, "%")
    return 0


if __name__ == "__main__":
    main()