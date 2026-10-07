import json
from pathlib import Path
from tabulate import tabulate

def generate_comparison_report():
    base_dir = Path(__file__).resolve().parent
    yolo_file = base_dir / "results" / "yolo" / "yolo_metrics.json"
    dfine_file = base_dir / "results" / "dfine" / "dfine_metrics.json"

    print("\n" + "=" * 70)
    print("      📊 HEAD-TO-HEAD VISION MODEL COMPARISON: YOLO11 vs D-FINE")
    print("=" * 70)

    yolo_data = None
    dfine_data = None

    if yolo_file.exists():
        with open(yolo_file, "r", encoding="utf-8") as f:
            yolo_data = json.load(f)
    else:
        print("[!] Note: YOLO11 metrics not found yet. Run 'python train_yolo.py' first.")

    if dfine_file.exists():
        with open(dfine_file, "r", encoding="utf-8") as f:
            dfine_data = json.load(f)
    else:
        print("[!] Note: D-FINE metrics not found yet. Run 'python train_dfine.py' first.")

    rows = [
        ["Model Name", 
         yolo_data.get("model_name", "YOLO11") if yolo_data else "Not Run", 
         dfine_data.get("model_name", "D-FINE") if dfine_data else "Not Run"],
        ["Architecture Family", 
         "One-Stage CNN + Attention", 
         "DETR / Box Distribution Refinement"],
        ["Overall mAP@50", 
         f"{yolo_data['mAP50']:.4f}" if yolo_data and yolo_data['mAP50'] > 0 else "Pending",
         f"{dfine_data['mAP50']:.4f}" if dfine_data and dfine_data['mAP50'] > 0 else "Pending"],
        ["Overall mAP@50:95 (Box Tightness)", 
         f"{yolo_data['mAP50_95']:.4f}" if yolo_data and yolo_data['mAP50_95'] > 0 else "Pending",
         f"{dfine_data['mAP50_95']:.4f}" if dfine_data and dfine_data['mAP50_95'] > 0 else "Pending"],
        ["Precision (P)", 
         f"{yolo_data['precision']:.4f}" if yolo_data and yolo_data['precision'] > 0 else "Pending",
         f"{dfine_data['precision']:.4f}" if dfine_data and dfine_data['precision'] > 0 else "Pending"],
        ["Recall (R)", 
         f"{yolo_data['recall']:.4f}" if yolo_data and yolo_data['recall'] > 0 else "Pending",
         f"{dfine_data['recall']:.4f}" if dfine_data and dfine_data['recall'] > 0 else "Pending"],
        ["F1-Score", 
         f"{yolo_data['f1_score']:.4f}" if yolo_data and yolo_data['f1_score'] > 0 else "Pending",
         f"{dfine_data['f1_score']:.4f}" if dfine_data and dfine_data['f1_score'] > 0 else "Pending"],
        ["Tap-Off Unit AP@50 (Common Class)", 
         f"{yolo_data['tap_off_unit_ap50']:.4f}" if yolo_data and yolo_data['tap_off_unit_ap50'] > 0 else "Pending",
         f"{dfine_data['tap_off_unit_ap50']:.4f}" if dfine_data and dfine_data['tap_off_unit_ap50'] > 0 else "Pending"],
        ["ZCT Sensor AP@50 (Rare Class)", 
         f"{yolo_data['zct_sensor_ap50']:.4f}" if yolo_data and yolo_data['zct_sensor_ap50'] > 0 else "Pending",
         f"{dfine_data['zct_sensor_ap50']:.4f}" if dfine_data and dfine_data['zct_sensor_ap50'] > 0 else "Pending"],
        ["Inference Latency", 
         f"{yolo_data['inference_latency_ms']:.2f} ms" if yolo_data and yolo_data['inference_latency_ms'] > 0 else "Pending",
         f"{dfine_data['inference_latency_ms']:.2f} ms" if dfine_data and dfine_data['inference_latency_ms'] > 0 else "Pending"],
    ]

    report_table = tabulate(rows, headers=["Metric / Parameter", "YOLO11", "D-FINE"], tablefmt="fancy_grid")
    print("\n" + report_table + "\n")

    # Save Markdown report
    report_md_path = base_dir / "results" / "comparison_report.md"
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 Single Line Diagram Vision Benchmark Report\n\n")
        f.write("```\n" + report_table + "\n```\n")
    print(f"[✓] Saved comparison report to: {report_md_path}")

if __name__ == "__main__":
    generate_comparison_report()
