import torch
import librosa
import torch.nn as nn
import logging
from neucodec import NeuCodec
from onnx_ops import OnnxResidualFSQ, OnnxISTFTHead


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Wrapper model")
        self.model = NeuCodec.from_pretrained("neuphonic/neucodec")
        self.model.eval()
        #self.model.half()
        logger.info("NeuCodec model loaded and set to eval mode")
        
    def forward(self, codes):
        logger.debug("Forward pass through model")
        return self.model.decode_code(codes)
    

@torch.inference_mode()
def main():
    logger.info("Starting ONNX export process")
    
    wrapper = Wrapper()
    wrapper_2 = Wrapper()
    
    logger.info("Initializing ONNX components")
    onnx_head = OnnxISTFTHead(
        dim=1024,
        n_fft=480*4,
        hop_length=480, 
    )
    onnx_fsq = OnnxResidualFSQ(
        dim=2048,
        levels=[4, 4, 4, 4, 4, 4, 4, 4],
        num_quantizers=1
    )
    
    logger.info("Loading state dict into ONNX FSQ")
    onnx_head.out.load_state_dict(wrapper.model.generator.head.out.state_dict())
    onnx_fsq.load_state_dict(wrapper.model.generator.quantizer.state_dict())
    print(wrapper.model.generator.quantizer.state_dict())
    
    logger.info("Encoding input audio")
    fsq_codes = wrapper.model.encode_code(librosa.ex("libri1"))

    import torchaudio
    original_decode = wrapper.model.decode_code(fsq_codes)
    torchaudio.save("original.wav", original_decode.squeeze(0), 24_000)
    print(f"Original decoding result shape: {original_decode.shape}")

    logger.info("Setting model attributes")
    wrapper.model.generator.head = onnx_head
    wrapper.model.generator.quantizer = onnx_fsq
    print(fsq_codes)
    fsq_post_emb = wrapper.model.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
    fsq_post_emb_2 = wrapper_2.model.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
    print(fsq_post_emb.shape, fsq_post_emb_2.shape)
    assert fsq_post_emb.shape == fsq_post_emb_2.shape
    assert torch.allclose(fsq_post_emb, fsq_post_emb_2)
    fsq_post_emb = fsq_post_emb.transpose(1, 2)
    fsq_post_emb_2 = fsq_post_emb_2.transpose(1, 2)
    fsq_post_emb = wrapper.model.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2) 
    fsq_post_emb_2 = wrapper_2.model.fc_post_a(fsq_post_emb_2.transpose(1, 2)).transpose(1, 2)
    assert fsq_post_emb.shape == fsq_post_emb_2.shape
    assert torch.allclose(fsq_post_emb, fsq_post_emb_2)
 
    new_decode = wrapper.model.decode_code(fsq_codes)
    print(f"new decoding result shape: {new_decode.shape}")
    torchaudio.save("new.wav", new_decode.squeeze(0), 24_000)
    
    logger.info("Exporting model to ONNX")
    print(f"Input shape: {fsq_codes.shape}")
    print(f"Input types: {fsq_codes.dtype}")
    onnx_program = torch.onnx.export(
        wrapper, 
        (fsq_codes,), 
        dynamo=True,
        dynamic_shapes={
            "codes": {0: "batch_size", 2: "sequence_length"}, 
        },
    )
    
    logger.info("Saving ONNX model to decoder.onnx")
    onnx_program.save("model.onnx")
    
    logger.info("ONNX export completed successfully")

if __name__ == "__main__":
    main()