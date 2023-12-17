from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from ultralytics import YOLO

def initialize_recognitor():  
    sam_checkpoint = './models/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device='cuda:1')

    predictor = SamPredictor(sam)

    model_name = './models/best.pt'
    model = YOLO(model_name)

    return (predictor, model)

if __name__ == '__main__':
    initialize_recognitor()