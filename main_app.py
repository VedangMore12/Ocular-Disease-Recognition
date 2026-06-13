import os
import textwrap
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from model.train_model import load_trained_model
from model.gradcam import generate_gradcam
from utils.preprocess import preprocess_image
from utils.recommend import get_recommendations
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


class OcularApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="flatly")  # clean, modern, medical-grade UI
        self.title("Ocular Disease Recognition and Wellness System")
        self.state("zoomed")
        self.configure(padx=20, pady=20)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_trained_model()
        self.model = self.model.to(self.device)
        self.image_paths = [None, None]
        self.tk_images = [None, None]
        self.heatmap_paths = [None, None]

        self.setup_ui()
        self.setup_styles()

    def setup_styles(self):
        """Define a clean, uniform font and color scheme for UI elements"""
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TFrame", background="#f8f9fa")
        style.configure("TLabelframe", background="#f8f9fa")
        style.configure("TLabelframe.Label", font=("Segoe UI", 13, "bold"))

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # LEFT PANEL
        left_frame = ttk.Frame(self, padding=25)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        left_frame.columnconfigure(0, weight=1)

        ttk.Label(
            left_frame,
            text="Ocular Disease Recognition",
            font=("Segoe UI", 22, "bold"),
            bootstyle="secondary"
        ).pack(pady=(10, 30))

        upload_frame = ttk.Frame(left_frame)
        upload_frame.pack(fill=X, pady=10)

        self.upload_btn1 = ttk.Button(upload_frame, text="Upload Image 1", bootstyle="info",
                                      command=lambda: self.upload_image(0))
        self.upload_btn1.pack(fill=X, pady=5)
        self.file_label1 = ttk.Label(upload_frame, text="No file selected", font=("Segoe UI", 9, "italic"))
        self.file_label1.pack(pady=(0, 10))

        self.upload_btn2 = ttk.Button(upload_frame, text="Upload Image 2", bootstyle="info",
                                      command=lambda: self.upload_image(1))
        self.upload_btn2.pack(fill=X, pady=5)
        self.file_label2 = ttk.Label(upload_frame, text="No file selected", font=("Segoe UI", 9, "italic"))
        self.file_label2.pack(pady=(0, 10))

        ttk.Separator(left_frame).pack(fill=X, pady=15)

        ttk.Label(left_frame, text="Patient Information", font=("Segoe UI", 14, "bold")).pack(pady=(10, 10))

        # AGE ENTRY
        age_frame = ttk.Frame(left_frame)
        age_frame.pack(pady=5, fill=X)
        ttk.Label(age_frame, text="Age:", font=("Segoe UI", 11)).pack(side="left", padx=(0, 8))
        self.age_var = ttk.StringVar()
        self.age_entry = ttk.Entry(age_frame, textvariable=self.age_var, width=12)
        self.age_entry.pack(side="left", fill=X, expand=True)
        self.age_entry.insert(0, "Enter Age")

        def clear_placeholder(event):
            if self.age_entry.get() == "Enter Age":
                self.age_entry.delete(0, "end")

        def restore_placeholder(event):
            if self.age_entry.get().strip() == "":
                self.age_entry.insert(0, "Enter Age")

        self.age_entry.bind("<FocusIn>", clear_placeholder)
        self.age_entry.bind("<FocusOut>", restore_placeholder)

        # SEX DROPDOWN
        sex_frame = ttk.Frame(left_frame)
        sex_frame.pack(pady=5, fill=X)
        ttk.Label(sex_frame, text="Sex:", font=("Segoe UI", 11)).pack(side="left", padx=(0, 8))
        self.sex_var = ttk.StringVar(value="Select")
        sex_menu = ttk.Combobox(sex_frame, textvariable=self.sex_var, values=["Male", "Female"], state="readonly",
                                width=12)
        sex_menu.pack(side="left", fill=X, expand=True)

        # CHECKBOXES
        self.diabetes_var = ttk.BooleanVar()
        self.hypertension_var = ttk.BooleanVar()
        ttk.Checkbutton(left_frame, text="Diabetes", variable=self.diabetes_var).pack(pady=5, anchor="w")
        ttk.Checkbutton(left_frame, text="Hypertension", variable=self.hypertension_var).pack(pady=5, anchor="w")

        ttk.Separator(left_frame).pack(fill=X, pady=20)

        # ACTION BUTTONS
        ttk.Button(left_frame, text="Run Diagnosis", bootstyle="success",
                   command=self.run_diagnosis).pack(pady=10, fill=X)
        ttk.Button(left_frame, text="Clear", bootstyle="danger",
                   command=self.clear_all).pack(pady=10, fill=X)

        # RIGHT PANEL
        right_panel = ttk.Frame(self, padding=20)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=4)
        right_panel.rowconfigure(1, weight=1)

        # IMAGE DISPLAY
        self.image_frame = ttk.Frame(right_panel)
        self.image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 15))

        self.image_labels = []
        for i in range(2):
            lbl = ttk.Label(self.image_frame, text=f"No Image {i + 1}",
                            font=("Segoe UI", 14, "italic"), anchor="center")
            lbl.grid(row=0, column=i, padx=10, sticky="nsew")
            self.image_frame.columnconfigure(i, weight=1)
            self.image_labels.append(lbl)

        # RESULTS FRAME
        results_frame = ttk.Labelframe(right_panel, text="Diagnosis Results", padding=15)
        results_frame.grid(row=1, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)

        self.result_text = ScrolledText(results_frame, wrap="word", font=("Consolas", 11), height=10)
        self.result_text.pack(expand=True, fill=BOTH)

    # ------------------ LOGIC METHODS ------------------

    def upload_image(self, index):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_paths[index] = file_path
            img = Image.open(file_path)
            img.thumbnail((350, 350))
            tk_img = ImageTk.PhotoImage(img)
            self.image_labels[index].configure(image=tk_img, text="")
            self.image_labels[index].image = tk_img
            (self.file_label1 if index == 0 else self.file_label2).configure(text=os.path.basename(file_path))

    def clear_all(self):
        self.image_paths = [None, None]
        for i, lbl in enumerate(self.image_labels):
            lbl.configure(image="", text=f"No Image {i + 1}")
            lbl.image = None
        self.file_label1.configure(text="No file selected")
        self.file_label2.configure(text="No file selected")
        self.result_text.delete("1.0", "end")
        self.age_entry.delete(0, "end")
        self.age_entry.insert(0, "Enter Age")
        self.sex_var.set("Select")
        self.diabetes_var.set(False)
        self.hypertension_var.set(False)

    def run_diagnosis(self):
        valid_paths = [p for p in self.image_paths if p]
        if not valid_paths:
            messagebox.showwarning("Warning", "Please upload at least one image first.")
            return

        try:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", "Analyzing images... please wait...\n")
            self.update_idletasks()

            all_probs = []
            for idx, path in enumerate(valid_paths):
                img_tensor = preprocess_image(path)
                img_batch = img_tensor.unsqueeze(0).to(self.device) if img_tensor.dim() == 3 else img_tensor.to(self.device)

                with torch.no_grad():
                    outputs = self.model(img_batch)
                    probs_tensor = torch.sigmoid(outputs).squeeze(0).cpu()
                all_probs.append(probs_tensor)

                try:
                    heatmap_path = generate_gradcam(self.model, path, probs_tensor)
                except Exception:
                    try:
                        pred_idx = int(torch.argmax(probs_tensor).item())
                        heatmap_path = generate_gradcam(self.model, path, pred_idx)
                    except Exception:
                        heatmap_path = None
                self.heatmap_paths[idx] = heatmap_path

                if heatmap_path and os.path.exists(heatmap_path):
                    heat_img = Image.open(heatmap_path)
                    heat_img.thumbnail((350, 350))
                    tk_heat = ImageTk.PhotoImage(heat_img)
                    self.image_labels[idx].configure(image=tk_heat, text="")
                    self.image_labels[idx].image = tk_heat

            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            avg_conf_val, pred_class_idx = torch.max(avg_probs, dim=0)
            avg_conf = float(avg_conf_val.item())
            pred_idx = int(pred_class_idx.item())

            disease_map = {
                0: "Normal (N)",
                1: "Diabetes (D)",
                2: "Glaucoma (G)",
                3: "Cataract (C)",
                4: "Age-related Macular Degeneration (A)",
                5: "Hypertension (H)",
                6: "Pathological Myopia (M)",
                7: "Other diseases/abnormalities (O)"
            }

            disease_name = disease_map.get(pred_idx, "Unknown")

            summary_map = {
                "Normal (N)": "Healthy retinal structure detected. No visible abnormalities.",
                "Diabetes (D)": "Signs of diabetic retinopathy such as microaneurysms or hemorrhages may be present.",
                "Glaucoma (G)": "Possible optic nerve cupping observed, may indicate increased intraocular pressure.",
                "Cataract (C)": "Lens opacity or cloudiness detected, which can reduce vision clarity.",
                "Age-related Macular Degeneration (A)": "Degeneration in the central retina area possibly linked to aging.",
                "Hypertension (H)": "Retinal vessel narrowing or hemorrhages suggestive of hypertensive retinopathy.",
                "Pathological Myopia (M)": "Elongated eyeball and retinal thinning suggest high myopia risk.",
                "Other diseases/abnormalities (O)": "Detected abnormalities not matching known disease categories."
            }
            summary = summary_map.get(disease_name, "Summary unavailable.")

            age = self.age_entry.get()
            sex = self.sex_var.get()
            diabetes = self.diabetes_var.get()
            hypertension = self.hypertension_var.get()
            recommendations = get_recommendations(disease_name, age, sex, diabetes, hypertension) or "No specific recommendations available."

            wrapped_summary = textwrap.fill(summary, width=100)
            wrapped_recommendations = textwrap.fill(recommendations, width=100)

            self.result_text.delete("1.0", "end")
            self.result_text.insert("end",
                f"Disease: {disease_name}\n"
                f"Confidence: {avg_conf:.2f}\n\n"
                f"Summary:\n{wrapped_summary}\n\n"
                f"Recommendations:\n{wrapped_recommendations}"
            )

            self.generate_visual_report(disease_name, avg_conf, wrapped_summary, wrapped_recommendations)

        except Exception as e:
            messagebox.showerror("Error", f"Diagnosis failed: {str(e)}")

    def generate_visual_report(self, disease_name, confidence, summary, recommendations):
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/Ocular_Report.pdf"

        c = canvas.Canvas(report_path, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 60, "Ocular Disease Recognition Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Predicted Disease: {disease_name}")
        c.drawString(50, height - 120, f"Confidence: {confidence:.2f}")
        c.drawString(50, height - 140, f"Patient Age: {self.age_entry.get()}")
        c.drawString(50, height - 160, f"Sex: {self.sex_var.get()}")

        y_pos = height - 300
        for i, path in enumerate(self.heatmap_paths):
            if path and os.path.exists(path):
                try:
                    c.drawImage(path, 50 + i * 250, y_pos, width=200, height=200)
                except Exception:
                    pass

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos - 30, "Summary:")
        text_obj = c.beginText(50, y_pos - 50)
        text_obj.setFont("Helvetica", 11)
        for line in summary.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos - 200, "Recommendations:")
        text_obj = c.beginText(50, y_pos - 220)
        text_obj.setFont("Helvetica", 11)
        for line in recommendations.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)

        c.save()
        messagebox.showinfo("Report Generated", f"Visual report saved to {report_path}")


if __name__ == "__main__":
    app = OcularApp()
    app.mainloop()
