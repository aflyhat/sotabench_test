
from torchbench.image_classification import ImageNet,ResNet34
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
import PIL

# Define the transforms need to convert ImageNet data to expected model input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,

    
])

# Run the benchmark
ImageNet.benchmark(
    model=ResNet34(pretrained=True),
    paper_model_name='ResNet34',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)