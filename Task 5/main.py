import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max(image.size)
    in_transform = transforms.Compose([
        transforms.Resize((max_size, int(max_size * image.size[0] / image.size[1]))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

content = load_image("content.jpg")
style = load_image("style.jpg")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers to extract
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content layer
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Content and style features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Target image
target = content.clone().requires_grad_(True).to(device)

# Style weights
style_weights = {'conv1_1': 1.0,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
for i in range(300):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_loss

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# Show and save result
final_img = target.clone().detach().cpu().squeeze()
final_img = final_img.permute(1, 2, 0)
final_img = final_img * torch.tensor((0.229, 0.224, 0.225)) + torch.tensor((0.485, 0.456, 0.406))
final_img = final_img.clamp(0, 1)
plt.imshow(final_img)
plt.axis('off')
plt.savefig("output.jpg")
plt.show()
