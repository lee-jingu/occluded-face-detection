from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def get_scores(model, dataloader, device) -> dict:
    sum_f1 = 0
    sum_precision = 0
    sum_recall = 0
    sum_accuracy = 0

    for img, label in dataloader:
        img = img.to(device)
        output = model(img)
        output = output > 0.5
        output = output.cpu()
        label = label.cpu().bool()

        sum_f1 += f1_score(label, output, average='samples', zero_division=0)
        sum_precision += precision_score(label, output, average='samples', zero_division=0)
        sum_recall += recall_score(label, output, average='samples', zero_division=0)
        sum_accuracy += accuracy_score(label, output)
    
    output = {
        'f1_score': sum_f1 / len(dataloader),
        'precision': sum_precision / len(dataloader),
        'recall': sum_recall / len(dataloader),
        'accuracy': sum_accuracy / len(dataloader)
    }

    return output