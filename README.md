🚀 Features

Full OTFS transmitter–receiver chain

Delay–Doppler channel simulation with delay & Doppler shifts

AWGN noise model

Lightweight AI tracking (Ridge regression or EWMA fallback)

Highly visual demo:

Transmitted DD Grid

Received DD Grid

Baseline Equalized DD Grid

AI-Predicted Equalized DD Grid

MSE vs SNR Curve

📦 Requirements

Install Python packages:

pip install numpy scipy matplotlib scikit-learn


If scikit-learn is not available, the script automatically switches to a fallback EWMA tracker.

▶️ How to Run the Simulation

Download or copy the file:

otfs_ai_demo.py


Open your terminal / command prompt:

cd path/to/your/folder


Run the script:

python otfs_ai_demo.py


The script will automatically open visual windows showing:

Transmitted DD-grid

Noisy Received DD-grid

Equalized DD-grid (baseline)

Equalized DD-grid (AI-based)

MSE vs SNR performance curve

📂 Generated Figures

You need to upload or save two main figures:

Figure 1: DD-Grid Visuals

This window contains:

Transmitted DD Grid

Received DD Grid

Baseline Equalized Grid

AI-Predicted Equalized Grid

Figure 2: MSE vs SNR Curve
🖼️ How to Save the Figures

When each plot appears:

Method 1 — Save directly from UI

Hover your mouse over the plot window.

Click the save icon (disk symbol) in the matplotlib toolbar.

Choose format:

PNG

JPEG

PDF

Save inside a folder named:

otfs_demo_outputs/
