
import torch
from tqdm import tqdm

def train (model, optimizer,  loss, train_dataloader, test_dataloader, epochs = 2, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    pbar = tqdm(range(epochs), colour = 'green')

    for epoch in pbar:
            
        total_train_loss = 0
        train_acc = 0
        total_val_loss = 0
        val_acc = 0
        avg_train_loss = 0
        avg_val_loss = 0
        avg_train_acc=0
        avg_val_acc = 0

        trained_samples = 0
        val_accs = []

        model.train()
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            total_train_loss += l.item()
            train_acc += (y_pred.argmax(1) == y).sum().item()
            trained_samples+= X.shape[0]

            avg_train_loss = total_train_loss/len(train_dataloader)
            avg_train_acc = train_acc/trained_samples
        
        validated_samples = 0

        with torch.no_grad():
            # pbar = tqdm(test_dataloader, colour = 'red')
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                l = loss(y_pred, y)
                total_val_loss += l.item()
                validated_samples += X.shape[0]
                val_acc += (y_pred.argmax(1) == y).sum().item()
                
            avg_val_loss = total_val_loss/len(test_dataloader)
            avg_val_acc = val_acc/validated_samples
            val_accs.append(avg_val_acc)
            # print(f"Validation Loss: {avg_val_loss :.3f}, Validation Accuracy: {avg_val_acc :.3f}")
        pbar.set_description(f"Epoch {epoch+1}, Loss: {avg_train_loss :.3f}, Train Accuracy: {avg_train_acc :.3f}, Validation Loss: {avg_val_loss :.3f}, Validation Accuracy: {avg_val_acc :.3f}")


    return model, avg_train_loss, avg_train_acc, avg_val_loss, max(val_accs)
        
        # wandb.log({'train_acc':avg_train_acc,"train_loss":avg_train_loss, "val_loss" : avg_val_loss, "val_acc": avg_val_acc})

def test(model,loss, test_dataloader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    test_acc = 0
    total_test_loss = 0
    test_samples = 0
    with torch.no_grad():
        pbar = tqdm(test_dataloader, colour = 'red')
        for i, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            l = loss(y_pred, y)
            total_test_loss += l.item()
            test_samples += X.shape[0]
            test_acc += (y_pred.argmax(1) == y).sum().item()
        avg_test_loss = total_test_loss/len(test_dataloader)
        avg_test_acc = test_acc/test_samples
        print(f"Test Loss: {avg_test_loss :.3f}, Test Accuracy: {avg_test_acc :.3f}")
        return avg_test_loss, avg_test_acc
