import os
import subprocess
import sys

def run_step(step_name, command):
    print(f"\n🚀 Running: {step_name}")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"❌ Error in {step_name}")
        sys.exit(1)
    else:
        print(f"✅ Completed: {step_name}")

def create_folders():
    folders = [
        "data/raw",
        "data/processed",
        "data/models"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("📁 Folder structure ready")

def main():
    print("\n==============================")
    print(" AI Predictive Maintenance Pipeline ")
    print("==============================\n")

    # Step 0: Setup folders
    create_folders()

    # Step 1: Simulate Data
    run_step("Simulating Data", "python src/simulate_data.py")

    # Step 2: Preprocess Data
    run_step("Preprocessing Data", "python src/preprocess.py")

    # Step 3: Train Model
    run_step("Training Model", "python src/train_model.py")

    # Step 4: Run Detection
    run_step("Running Detection", "python src/detect.py")

    print("\n🎯 Pipeline Execution Completed Successfully!")

    print("\n📊 To launch dashboard run:")
    print("👉 streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    main()