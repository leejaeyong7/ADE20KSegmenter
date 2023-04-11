# Light-weight wrapper repository for running ADE20K semantic segmentation

# License
The license for the model and the weights are entirely attributed to the original authors of the [ADE20K semantic segmentation repository](https://github.com/CSAILVision/semantic-segmentation-pytorch)

# Installation
```bash
pip install git+http://github.com/leejaeyong7/ADE20KSegmenter
```

# Example Usage
```python
import torch
import torchvision.transforms as transforms
from ade20k_segmenter import ADE20KSegmenter
from PIL import Image

device = torch.device('cuda')

seg_model = ADE20KSegmenter().to(device)
seg_model.eval()

crop_size = 800
image_size = 384
image_file = 'test.png'

trans =  transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.Resize(image_size, interpolation=Image.BILINEAR),
    transforms.ToTensor()
])
norm_trans = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

image = Image.open(image_file)
cropped = trans(image)
normalized = norm_trans(cropped)

normalized = normalized[:3].unsqueeze(0).to(device)

with torch.no_grad():
    logits = seg_model(normalized)
```