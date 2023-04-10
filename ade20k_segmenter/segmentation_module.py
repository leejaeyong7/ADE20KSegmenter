from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from pathlib import Path
import torch
import torch.nn as nn

class ADE20KSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        weight_path = Path(__file__).parent / 'pretrained_weights'
        weight_path.mkdir(exist_ok=True, parents=True)

        # 
        enc_weight_path = weight_path / 'encoder_epoch_50.pth'
        dec_weight_path = weight_path / 'decoder_epoch_50.pth'

        # download weights from server
        # encoder path: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101-upernet/encoder_epoch_50.pth
        # decoder path: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101-upernet/decoder_epoch_50.pth
        from tqdm import tqdm
        import requests

        if not enc_weight_path.exists():
            enc_url = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101-upernet/encoder_epoch_50.pth"
            enc_response = requests.get(enc_url, stream=True)
            length = None
            if 'Content-length' in enc_response.headers:
                length = int(enc_response.headers['Content-length'])

            with open(enc_weight_path, "wb") as handle:
                for data in tqdm(enc_response.iter_content(), total=length, desc='fetching encoder weights'):
                    handle.write(data)

        if not dec_weight_path.exists():
            dec_url = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101-upernet/decoder_epoch_50.pth"
            dec_response = requests.get(dec_url, stream=True)
            length = None
            if 'Content-length' in dec_response.headers:
                length = int(dec_response.headers['Content-length'])
            with open(dec_weight_path, "wb") as handle:
                for data in tqdm(dec_response.iter_content(), total=length, desc='fetching decoder weights'):
                    handle.write(data)

        net_encoder = ModelBuilder.build_encoder(
            arch='resnet101',
            fc_dim=2048,
            weights=str(enc_weight_path))

        net_decoder = ModelBuilder.build_decoder(
            arch='upernet',
            fc_dim=2048,
            num_class=150,
            weights=str(dec_weight_path),
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)

        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module = segmentation_module

    def forward(self, x):
        '''
        Assumes that the input is a tensor of shape (N, C, H, W)
        '''
        # normalize image
        singleton_batch = {'img_data': x}
        output_size = x.shape[-2:]
        return self.segmentation_module(singleton_batch, segSize=output_size)

