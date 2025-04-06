
# 🎨 Air Canvas with Depth-Anything V2

**Draw in the air using hand gestures + AI-powered depth estimation.**  
Built using MediaPipe, OpenCV, and Depth-Anything V2.  
This is not just your regular canvas — this one's hands-free and brainy 😎

---

## 🚀 Features

- ✋ Real-time hand tracking with MediaPipe
- 🖌️ Draw using your fingertip in the air
- 🧠 AI-based depth estimation using Depth-Anything V2 (Meta)
- 🎯 Depth-aware drawing (future: 3D enhancements maybe?)
- 🎨 Cool color palette and thickness selection (coming soon)

---

## 🛠️ Installation

### 1. Clone the repo

\`\`\`bash
git clone https://github.com/Rcidshacker/Air-canvas-new.git
cd Air-canvas-new
\`\`\`

### 2. Create a virtual environment (recommended)

\`\`\`bash
python -m venv .depthv2
\`\`\`

### 3. Activate it

- **Windows:**  
  \`.\.depthv2\Scripts\activate\`

- **Mac/Linux:**  
  \`source .depthv2/bin/activate\`

### 4. Install dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 5. Download model weights

> ⚠️ GitHub doesn’t allow large files (>100 MB), so you’ll need to manually download model weights:

- \`depth_anything_v2_vits.pth\` → Put it in \`model_weights/\`

Get it from [here](https://github.com/isl-org/Depth-Anything#model-weights) (Depth-Anything repo)

---

## ▶️ Run the App

\`\`\`bash
python air_canvas.py
\`\`\`

Make sure your webcam is connected!

---

## 📁 Project Structure

\`\`\`
├── air_canvas.py               # Main application
├── model_weights/              # Folder for model .pth file
├── utils/                      # Utility functions for depth + drawing
├── .depthv2/                   # Your virtual environment (ignored in git)
├── requirements.txt
└── README.md
\`\`\`

---

## ⚡ Future Ideas

- Add gesture-controlled tools (eraser, color picker)
- Save your masterpiece to disk
- Use depth for layered/3D drawing
- Web version with WebRTC?

---

## 🙋‍♂️ Author

Made with ❤️ by **Ruchit**  
Feel free to contribute, suggest or vibe with the code.  
[LinkedIn](https://www.linkedin.com/in/rcidshacker/) | [GitHub](https://github.com/Rcidshacker)
