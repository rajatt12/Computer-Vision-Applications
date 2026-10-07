from ultralytics import YOLO
import sys

def main():
    model_path = "best.pt"
    image_path = "../Pretrained/circuit-vision/images/substation-with-single-transformer.png"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {image_path}")
    results = model.predict(source=image_path, conf=0.5, save=True)
    
    print("\nDetection complete! Results saved in: runs/obb/predict/")

if __name__ == "__main__":
    main()
