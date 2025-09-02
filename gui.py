import tkinter as tk
from tkinter import filedialog, ttk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# Load trained model
model = load_model("pneumonia_model.h5")

# Prediction function
def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "🛑 Pneumonia Detected" if prediction[0][0] > 0.5 else "✅ Normal"

    result_label.config(text=result, fg="red" if "Pneumonia" in result else "green")

    img_display = Image.open(file_path)
    img_display = img_display.resize((300, 300))
    img_display = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_display)
    image_label.image = img_display

# --- Main Window Setup ---
root = tk.Tk()
root.title("DeepLung - Pneumonia Detection AI")
root.geometry("900x700")
root.configure(bg="#f7faff")

style = ttk.Style()
style.configure("TNotebook", background="#f7faff", tabposition='n')
style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 12, "bold"))

# --- Tabbed UI ---
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# --- Detection Tab ---
detection_tab = tk.Frame(notebook, bg="#f0faff")
notebook.add(detection_tab, text="🧪 Detection")

# Header
header = tk.Label(detection_tab, text="DeepLung: Pneumonia Detector", font=("Segoe UI", 26, "bold"), bg="#f0faff", fg="#003049")
header.pack(pady=(30, 10))

# Subheader
subheader = tk.Label(detection_tab, text="Upload a chest X-ray image to predict pneumonia using AI", 
                     font=("Roboto", 14), bg="#f0faff", fg="#333")
subheader.pack(pady=(0, 20))

# Upload Button
upload_btn = tk.Button(detection_tab, text="📤 Upload X-Ray", font=("Segoe UI", 14, "bold"),
                       bg="#00b4d8", fg="white", activebackground="#0077b6",
                       padx=20, pady=10, borderwidth=0, command=predict_image)
upload_btn.pack(pady=10)

# Image Display
image_label = tk.Label(detection_tab, bg="#f0faff")
image_label.pack(pady=10)

# Prediction Result
result_label = tk.Label(detection_tab, text="", font=("Segoe UI", 20, "bold"), bg="#f0faff")
result_label.pack(pady=20)

# Footer
footer = tk.Label(detection_tab, text="🌡️ Built for Healthcare AI Innovation", font=("Roboto", 12), bg="#f0faff", fg="#888")
footer.pack(side="bottom", pady=15)

# --- Credits Tab ---
credits_tab = tk.Frame(notebook, bg="#fff6f6")
notebook.add(credits_tab, text="👨‍💻 Credits")

credits_title = tk.Label(credits_tab, text="Credits", font=("Segoe UI", 24, "bold"), bg="#fff6f6", fg="#d62828")
credits_title.pack(pady=(40, 10))

dev_info = tk.Label(credits_tab, text=(
    "Developed as part of a Semester Project\n\n"
    "🧠 Model: CNN with TensorFlow\n"
    "🎨 GUI: Tkinter\n\n"
    "👨‍💻 Developer: A passionate app dev \n"
    "Thank you for exploring this project!"
), font=("Roboto", 14), bg="#fff6f6", fg="#333", justify="left")
dev_info.pack(pady=10)

# Launch the app
root.mainloop()
