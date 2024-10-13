from ultralytics import YOLO
import torch
def predict(file_name):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = YOLO(r"best.pt").to(device)
    model.predict(source = file_name , save = True , conf=0.5 , project = 'pred',name ="anno" , iou =0.1) 
    