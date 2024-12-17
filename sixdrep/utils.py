from torch.utils import data
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
sixdreptransform = transforms.Compose([transforms.Resize(224 + 32),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])