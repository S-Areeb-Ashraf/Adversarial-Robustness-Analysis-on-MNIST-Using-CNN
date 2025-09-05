import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad=data_grad.sign()
    perturbed_image=image + epsilon * sign_data_grad
    perturbed_image=torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def generate_adversarial_example(model, data, target, epsilon):
    model.eval()

    data.requires_grad=True
    output=model(data)
    init_pred=output.max(1, keepdim=True)[1]
    if init_pred.item() != target.item():
        return data, data, init_pred

    loss=F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    output = model(perturbed_data)
    final_pred = output.max(1, keepdim=True)[1]
    
    return data, perturbed_data, final_pred