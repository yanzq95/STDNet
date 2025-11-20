import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class GIFCreatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GIF")
        self.root.geometry("700x700")

        self.images = []
        self.image_paths = []

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="GIF", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 选择图片按钮
        self.select_btn = tk.Button(self.root, text="选择图片", command=self.select_images,
                                    bg="#4CAF50", fg="white", font=("Arial", 12))
        self.select_btn.pack(pady=5)

        # 图片列表框架
        list_frame = tk.LabelFrame(self.root, text="已选图片", padx=10, pady=10)
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # 滚动条
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 图片列表
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # 控制按钮框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.up_btn = tk.Button(control_frame, text="上移", command=self.move_up)
        self.up_btn.grid(row=0, column=0, padx=5)

        self.down_btn = tk.Button(control_frame, text="下移", command=self.move_down)
        self.down_btn.grid(row=0, column=1, padx=5)

        self.remove_btn = tk.Button(control_frame, text="移除", command=self.remove_image)
        self.remove_btn.grid(row=0, column=2, padx=5)

        # 参数设置框架
        params_frame = tk.LabelFrame(self.root, text="GIF参数", padx=10, pady=10)
        params_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(params_frame, text="帧率(fps):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.fps_var = tk.IntVar(value=25)  # 默认改为25fps
        tk.Entry(params_frame, textvariable=self.fps_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(params_frame, text="循环次数(0为无限循环):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.loop_var = tk.IntVar(value=0)
        tk.Entry(params_frame, textvariable=self.loop_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # 创建GIF按钮
        self.create_btn = tk.Button(self.root, text="创建GIF", command=self.create_gif,
                                    bg="#2196F3", fg="white", font=("Arial", 12))
        self.create_btn.pack(pady=10)

        # 状态标签
        self.status_label = tk.Label(self.root, text="准备就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_images(self):
        filetypes = (
            ('图片文件', '*.png *.jpg *.jpeg *.bmp *.tiff'),
            ('所有文件', '*.*')
        )

        filenames = filedialog.askopenfilenames(title="选择图片", filetypes=filetypes)

        if filenames:
            for filename in filenames:
                if filename not in self.image_paths:
                    self.image_paths.append(filename)
                    self.listbox.insert(tk.END, os.path.basename(filename))

            self.status_label.config(text=f"已选择 {len(self.image_paths)} 张图片")

    def move_up(self):
        selected = self.listbox.curselection()
        if selected and selected[0] > 0:
            index = selected[0]
            # 移动列表中的项目
            self.image_paths[index], self.image_paths[index - 1] = self.image_paths[index - 1], self.image_paths[index]
            # 更新列表框
            self.update_listbox()
            self.listbox.select_set(index - 1)

    def move_down(self):
        selected = self.listbox.curselection()
        if selected and selected[0] < len(self.image_paths) - 1:
            index = selected[0]
            # 移动列表中的项目
            self.image_paths[index], self.image_paths[index + 1] = self.image_paths[index + 1], self.image_paths[index]
            # 更新列表框
            self.update_listbox()
            self.listbox.select_set(index + 1)

    def remove_image(self):
        selected = self.listbox.curselection()
        if selected:
            index = selected[0]
            self.image_paths.pop(index)
            self.update_listbox()
            self.status_label.config(text=f"已选择 {len(self.image_paths)} 张图片")

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

    def create_gif(self):
        if len(self.image_paths) < 2:
            messagebox.showerror("错误", "请至少选择两张图片！")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=(("GIF文件", "*.gif"), ("所有文件", "*.*"))
        )

        if not output_path:
            return

        try:
            # 打开所有图片
            images = []
            for path in self.image_paths:
                img = Image.open(path)
                images.append(img.copy())

            # 保存为GIF
            fps = self.fps_var.get()
            delay = int(1000 / fps)  # 将fps转换为毫秒延迟
            loop = self.loop_var.get()

            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=delay,
                loop=loop
            )

            messagebox.showinfo("成功", f"GIF已成功创建！\n保存位置: {output_path}\n帧率: {fps}fps")
            self.status_label.config(text="GIF创建成功")

        except Exception as e:
            messagebox.showerror("错误", f"创建GIF时出错:\n{str(e)}")
            self.status_label.config(text="创建GIF时出错")


if __name__ == "__main__":
    root = tk.Tk()
    app = GIFCreatorApp(root)
    root.mainloop()