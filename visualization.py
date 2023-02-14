from torch import nn
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# set global seed
torch.manual_seed(0)

# Fix for strange error with Tensorboard
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=3, padding="valid"
        )
        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=7, kernel_size=3, padding="valid"
        )
        self.conv3 = nn.Conv2d(
            in_channels=7, out_channels=5, kernel_size=3, padding="valid"
        )
        self.conv4 = nn.Conv2d(
            in_channels=5, out_channels=1, kernel_size=3, padding="valid"
        )

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear1 = nn.Linear(in_features=400, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        e = self.flatten(x)
        x = self.linear1(e)

        return x, e


def plot_confusion_matrix(model: nn.Module, step: int = 0):

    # confusion matrix vector
    y_pred = []
    y_true = []

    for x, y in dataloader_val:
        y_hat, _ = model(x)
        y_idx = torch.argmax(y_hat, 1)

        y_pred.extend(list(y_idx.numpy()))
        y_true.extend(list(y.numpy()))

    cm = confusion_matrix(y_true, y_pred)

    figure = plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.ylabel("Predictions")
    plt.xlabel("Truth")
    plt.savefig(f"./logs/confusion_matrix_step={step}.png")
    plt.close()


def generate_projection(model: nn.Module, model_path: str, dataloader: DataLoader):

    # Create embedding visualization
    model.load_state_dict(torch.load(model_path))

    outputs = []
    data = []
    images = []

    for x, y in dataloader:

        # remove the gradient tape
        with torch.no_grad():
            _, embedding = model(x)
            outputs.append(embedding)

        data.append(y)
        images.append(x)

    # stack the output embeddings
    matrix = torch.vstack(outputs)
    metadata = torch.hstack(data)
    label_img = torch.vstack(images)

    # Tensorboard is broken unless this is done!
    mean, std, _ = torch.mean(matrix), torch.std(matrix), torch.var(matrix)
    matrix = (matrix - mean) / std
    matrix = matrix[0:1000, :]
    metadata = metadata[0:1000]
    label_img = label_img[0:1000, :, :, :]

    writer.add_embedding(
        matrix,
        metadata=metadata,
        label_img=label_img,
        global_step=0,
    )


def saliency_map(model: nn.Module, model_path: str, dataloader: DataLoader):

    # configuration
    n_columns = 10

    # Create embedding visualization
    model.load_state_dict(torch.load(model_path))

    for x, y in dataloader:

        for ii in range(1, n_columns):

            # calculate gradients w.r.t. input from output
            x_src = x[ii]
            x_hat = x[ii]  # create a copy of the tensor
            x_hat.requires_grad = True  # set gradient tape to True
            y_hat, _ = model(x_hat)  # forward prop
            target = y[ii]  # target class
            y_hat[0][target].backward()  # backward prop only on target class!

            # generate subplots
            ax1 = plt.subplot(2, n_columns, ii)
            ax2 = plt.subplot(2, n_columns, ii + n_columns)

            # normalize the saliency plot
            img_saliency = x_hat.grad.squeeze()
            mean, std, _ = (
                torch.mean(img_saliency),
                torch.std(img_saliency),
                torch.var(img_saliency),
            )

            # create images for plotting
            img_saliency = (img_saliency - mean) / std
            img_saliency[img_saliency < 0] = 0  # remove values less than 0
            img_source = x_src.squeeze().detach().numpy()

            # plot images
            ax1.imshow(img_saliency, cmap=plt.cm.viridis, aspect="auto")
            ax2.imshow(img_source, cmap=plt.cm.viridis, aspect="auto")

            ax1.axis("off")
            ax2.axis("off")


def get_dataloaders():
    # dataset/dataloader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset_train = MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_val = MNIST(root="./data", train=False, download=True, transform=transform)

    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True
    )
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=batch_size, shuffle=False
    )

    return dataloader_train, dataloader_val


def training_iteration(model: nn.Module, loss_fn, optimizer):

    # training interation
    train_losses = []
    for x, y in tqdm(
        dataloader_train,
        desc="training",
        colour="blue",
        position=1,
    ):
        y_hat, _ = model(x)
        loss = loss_fn(y_hat, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.average(train_losses)
    writer.add_scalar("Loss/train", avg_train_loss, ii)

    return avg_train_loss


def eval_iteration():

    # evaluation iteration
    eval_losses = []
    for x, y in tqdm(
        dataloader_val,
        desc="evaluation",
        colour="teal",
        position=2,
    ):
        with torch.no_grad():
            y_hat, _ = model(x)
            loss = loss_fn(y_hat, y.long())
            loss.backward()
            eval_losses.append(loss.item())

    avg_eval_loss = np.average(eval_losses)
    writer.add_scalar("Loss/eval", avg_eval_loss, ii)

    return avg_eval_loss


if __name__ == "__main__":

    # configurable params
    n_epochs = 10
    min_loss = np.inf
    batch_size = 16
    best_model_path = "./save/best_model.pkl"

    # get the dataloaders
    dataloader_train, dataloader_val = get_dataloaders()

    # tensorboard
    writer = SummaryWriter("./logs")

    # hyperparameters
    model = Model()
    optimizer = Adam(params=model.parameters(), lr=0.0001)
    loss_fn = CrossEntropyLoss()

    # save off model
    writer.add_graph(model, torch.rand(1, 1, 28, 28))

    # # perform training on the data
    # for ii in tqdm(range(0, n_epochs), desc="epochs", colour="green", position=0):

    #     avg_train_loss = training_iteration(
    #         model=model, loss_fn=loss_fn, optimizer=optimizer
    #     )

    #     avg_eval_loss = eval_iteration(
    #         model=model, loss_fn=loss_fn, optimizer=optimizer
    #     )

    #     # printout
    #     print(
    #         f"Epoch: {ii}/{n_epochs}  Train Loss: {avg_train_loss:.3f}  Eval Loss: {avg_eval_loss:.3f}"
    #     )

    #     # save off best model
    #     if avg_eval_loss < min_loss:
    #         min_loss = avg_eval_loss
    #         torch.save(model.state_dict(), best_model_path)
    #         plot_confusion_matrix(model=model, step=ii)  # save off confusion matrix
    #         print(f"Saving best model with evaluation loss of: {avg_eval_loss}")

    # generate projetion
    # generate_projection(model, model_path=best_model_path, dataloader=dataloader_val)
    saliency_map(model=model, model_path=best_model_path, dataloader=dataloader_val)

    writer.close()
