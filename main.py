from predictor.ml_predictor import predict_regions
import time

def main():
    while True:
        preds = predict_regions()
        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)

        print("\n=== SEISMIC RISK (ML MODEL) ===")
        for r, p in ranked:
            print(f"{r:20s} -> P={p:.3f}")

        time.sleep(60)

if __name__ == "__main__":
    main()
