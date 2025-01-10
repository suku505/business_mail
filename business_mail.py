import tkinter as tk
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap


def build_prompt(user_query):
    sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <>
{}
<>

{}[/INST]"""
    return template.format(sys_msg, user_query)


def addList(text):
    # Infer with prompt without any additional input
    user_inputs = {
        "user_query": f"{text}の内容をビジネスメールで",
    }
    prompt = build_prompt(**user_inputs)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

    ListBox1.delete(0, tk.END)
    ListBox2.delete(0, tk.END)
    mysay = "変換内容: " + text
    print(mysay)
    ListBox1.insert(tk.END, mysay)
    Entry1.delete(0, tk.END)
    ListBox2.insert(tk.END, "変換中です。お待ち下さい。")

    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=700,
        temperature=1,
        top_p=0.95,
        do_sample=True,
    )
    out = tokenizer.decode(
        tokens[0][input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    ListBox2.delete(0, tk.END)
    add_item_with_wrap(ListBox2, out)


def add_item_with_wrap(listbox, text):
    width = listbox["width"] - 28

    # 改行を保持しながら処理
    for paragraph in text.split("\n"):  # 元の改行ごとに分割
        if paragraph.strip():
            wrapped_text = textwrap.fill(paragraph, width=width)
            for line in wrapped_text.split("\n"):  # 自動改行された行を追加
                listbox.insert(tk.END, line)
        else:
            listbox.insert(tk.END, "")  # 空行をそのまま追加


tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KillerWhale")
model = AutoModelForCausalLM.from_pretrained(
    "DataPilot/ArrowPro-7B-KillerWhale",
    torch_dtype="auto",
)
model.eval()

root = tk.Tk()
root.title("ビジネスメール変換")
root.geometry("800x500")

Static1 = tk.Label(text="▼ 変換内容 ▼")
Static1.pack()
Entry1 = tk.Entry(width=50)

Button1 = tk.Button(
    text="送信", width=50, command=lambda: addList(Entry1.get())
)  # 関数に引数を渡す場合は、commandオプションとlambda式を使う
Button1.pack()


SubTitle1 = tk.Label(text="▼ 変換前内容 ▼")
SubTitle1.place(x=10, y=100, width=350)
SubTitle2 = tk.Label(text="▼ 変換後内容 ▼")
SubTitle2.place(x=360, y=100, width=350)

ListBox1 = tk.Listbox(width=55, height=20)
ListBox1.pack(side=tk.LEFT, padx=10)
ListBox2 = tk.Listbox(width=70, height=20)
ListBox2.pack(side=tk.RIGHT, padx=10)

root.mainloop()
