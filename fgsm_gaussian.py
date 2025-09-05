import torch
import torch.nn.functional as F

def fgsm_gaussian_attack(image, epsilon):
    gaussian_noise=torch.randn_like(image) * epsilon
    
    perturbed_image=image + gaussian_noise
    perturbed_image=torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_gaussian_adversarial_example(model, data, target, epsilon):
    model.eval()
    with torch.no_grad():
        output=model(data)
        init_pred=output.max(1, keepdim=True)[1]
    
    if init_pred.item()!=target.item():
        return data,data,init_pred
    
    perturbed_data=fgsm_gaussian_attack(data, epsilon)
    with torch.no_grad():
        output=model(perturbed_data)
        final_pred=output.max(1, keepdim=True)[1]
    
    return data,perturbed_data,final_pred