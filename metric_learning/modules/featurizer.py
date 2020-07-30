import pretrainedmodels
import torch.nn as nn
import torch.hub as hub
from torch.utils.checkpoint import checkpoint_sequential


class EmbeddedFeatureWrapper(nn.Module):
    """
    Wraps a base model with embedding layer modifications.
    """
    def __init__(self,
                 feature,
                 input_dim,
                 output_dim,
                 chunks=2):
        super(EmbeddedFeatureWrapper, self).__init__()

        self.chunks = chunks
        self.feature = feature
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.remap = None
        if input_dim != output_dim:
            self.remap = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, images):
        # x = checkpoint_sequential(self.feature, self.chunks, images)
        x = self.feature(images)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)

        if self.remap:
            x = self.remap(x)

        x = nn.functional.normalize(x, dim=1)

        return x

    def __str__(self):
        return "{}_{}".format(self.feature.name, str(self.embed))


def resnet50(output_dim):
    """
    resnet50 variant with `output_dim` embedding output size.
    """
    # basemodel = pretrainedmodels.__dict__["resnet50"](num_classes=1000)
    basemodel = hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    model = nn.Sequential(
        basemodel.conv1,
        basemodel.bn1,
        basemodel.relu,
        basemodel.maxpool,

        basemodel.layer1,
        basemodel.layer2,
        basemodel.layer3,
        basemodel.layer4
    )
    model.name = "resnet50"
    featurizer = EmbeddedFeatureWrapper(feature=model, input_dim=2048, output_dim=output_dim)
    featurizer.input_space = 'RGB'
    featurizer.input_range = [0, 1]
    featurizer.input_size = [3, 224, 224]
    featurizer.std = [0.229, 0.224, 0.225]
    featurizer.mean = [0.485, 0.456, 0.406]

    return featurizer

def se_resnext101_32x4d(output_dim):
    """
    se_resnext101_32x4d variant with `output_dim` embedding output size.
    """
    basemodel = pretrainedmodels.__dict__["se_resnext101_32x4d"](num_classes=1000)

    model = nn.Sequential(
        basemodel.layer0,
        basemodel.layer1,
        basemodel.layer2,
        basemodel.layer3,
        basemodel.layer4
    )
    model.name = "se_resnext101_32x4d"
    featurizer = EmbeddedFeatureWrapper(feature=model, input_dim=2048, output_dim=output_dim)
    featurizer.input_space = 'RGB'
    featurizer.input_range = [0, 1]
    featurizer.input_size = [3, 224, 224]
    featurizer.std = [0.229, 0.224, 0.225]
    featurizer.mean = [0.485, 0.456, 0.406]

    return featurizer

def resnest50(output_dim):
    """
    resnest50 variant with `output_dim` embedding output size.
    """
    basemodel = hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

    model = nn.Sequential(
        basemodel.conv1,
        basemodel.bn1,
        basemodel.relu,
        basemodel.maxpool,

        basemodel.layer1,
        basemodel.layer2,
        basemodel.layer3,
        basemodel.layer4
    )
    model.name = "resnest50"
    featurizer = EmbeddedFeatureWrapper(feature=model, input_dim=2048, output_dim=output_dim)
    featurizer.input_space = 'RGB'
    featurizer.input_range = [0, 1]
    featurizer.input_size = [3,224,224]
    featurizer.std = [0.229, 0.224, 0.225]
    featurizer.mean = [0.485, 0.456, 0.406]

    return featurizer
