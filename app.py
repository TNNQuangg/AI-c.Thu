import tkinter as tk
from tkinter import messagebox
from utils import load_model
from vectorizer import text_to_vector
from random_forest import predict_random_forest

MODEL_FILE = "trained_model.pkl"
tree, vocab = load_model(MODEL_FILE)

def on_click_check():
    user_input = text_entry.get().strip()
    
    if not user_input:
        messagebox.showwarning("Vui lòng nhập câu cần kiểm tra!")
        return
    
    result_label.config(text="Đang phân tích... ⏳", fg="gray")
    
    root.after(500, lambda: process_and_display(user_input))

def process_and_display(user_input):
    # Tiền xử lý và vector hóa câu nhập vào
    sample_vector = text_to_vector(user_input, vocab, remove_stopwords=True, use_bigrams=True)
    
    # Dự đoán
    prediction, sample_probs = predict_random_forest(tree, sample_vector, neutral_threshold=0.5)
    prediction = prediction[0]
    
    label_map = {0: "Negative (Tiêu cực)", 1: "Neutral (Trung lập)", 2: "Positive (Tích cực)"}
    
    # Lấy 3 thông số xác suất
    neg_prob = sample_probs[0][0]
    neu_prob = sample_probs[0][1]
    pos_prob = sample_probs[0][2]
    
    # Cập nhật kết quả lên giao diện
    result_text = (
        f"--- XÁC SUẤT ---\n"
        f"Negative : {neg_prob:.3f}; "
        f"Neutral  : {neu_prob:.3f}; "
        f"Positive : {pos_prob:.3f}\n\n"
        f"=> KẾT QUẢ DỰ ĐOÁN: {label_map.get(prediction, 'Unknown')}"
    )
    
    color = "blue" if prediction == 2 else "red" if prediction == 0 else "black"
    result_label.config(text=result_text, fg=color)

root = tk.Tk()
root.title("App Phân Loại Cảm Xúc Văn Bản")
root.geometry("450x350")
root.eval('tk::PlaceWindow . center') 

instruction_label = tk.Label(root, text="Nhập câu cần phân tích:", font=("Arial", 12, "bold"))
instruction_label.pack(pady=(20, 5))

text_entry = tk.Entry(root, width=50, font=("Arial", 12))
text_entry.pack(pady=5)

check_button = tk.Button(root, text="Kiểm tra", command=on_click_check, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
check_button.pack(pady=15)

result_label = tk.Label(root, text="", font=("Courier New", 12, "bold"), justify="left")
result_label.pack(pady=10)

root.mainloop()