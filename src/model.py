import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.optim import lr_scheduler


class model:
    def __init__(self):
        self.data_dir = '../data'
        self.batch_size = 64
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 5
        self.input_size = 224

    def __call__(self):
        self.__load_data()
        self.__initialize_model()

    def __load_data(self):
        # create a dictionary with the corresponding transformation to both hot and nonhot. For training
        # augment data and normalize it. For nonhot just norm.
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        # create dictionary {train: trainingData, val: ValidationData}
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x])
                                    for x in ['train', 'val']
        }
        # Create training and validation dataloaders
        self.dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x
            in ['train', 'val']}

    def __optimizer(self):
        model_ft = self.model.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.model.parameters()
        print("Params to learn:")
        for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return self.optimizer

    def train_handler(self):
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        # Train and evaluate
        self.model, self.hist = self.__train(criterion, self.__optimizer(), self.exp_lr_scheduler,
                                             num_epochs=self.num_epochs)

    def __train(self, criterion, optimizer, scheduler, num_epochs=5):
        start_training_time = time.time()
        self.val_acc_history = []
        best_model_params = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-'*10)
            # first training phase, the validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() # set model to training mode
                else:
                    self.model.eval() # set model to eval mode
                current_loss = 0.0
                current_corrects = 0
                # iterate over data
                for inputs, labels in self.dataloaders_dict[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # reset parameter gradients
                    optimizer.zero_grad()


                    with torch.set_grad_enabled(phase == 'train'):
                        # get model outputs and calculate loss
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        ## perform backpropagation
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # compute statistics
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = current_loss/len(self.dataloaders_dict[phase].dataset)
                epoch_acc = current_corrects.double() / len(self.dataloaders_dict[phase].dataset)
                print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

                # save best model if performs better than current best model in evaluation
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_params = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    self.val_acc_history.append(epoch_acc)
            print()
        time_elapse = time.time() - start_training_time
        print(f'Training completed in {time_elapse//60} minutes and {time_elapse%60} seconds')

        # load best model parameters
        self.model.load_state_dict(best_model_params)
        return self.val_acc_history



    def __initialize_model(self):
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2) # default requires_grad True

m = model()
m()