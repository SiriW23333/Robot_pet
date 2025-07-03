import tkinter as tk
from tkinter import ttk, messagebox
import random
from datetime import datetime
import os
import json
from PIL import Image, ImageTk

class PetUI:
    def __init__(self, initial_affection=0):
        self.root = tk.Tk()
        self.root.title("AI宠物交互界面 - RDK X5")
        self.root.geometry("700x550")  # 修改为700x550
        self.root.configure(bg='#f0f8ff')
        
        # 设置窗口居中
        self.center_window()
        
        # 设置窗口图标和样式
        self.root.resizable(True, True)
        self.root.minsize(650, 500)  # 修改最小尺寸
        
        # 配置ttk样式
        self.setup_styles()
        
        # 初始化数据（不使用数据库，使用内存变量）
        self.affection = initial_affection
        self.last_checkin = None
        self.last_task_date = None
        self.task_completed = 0
        # 初始化任务字典
        self.task_dict = self.get_default_tasks()
        
        # 当前任务和完成状态
        self.current_task = None
        self.task_completed_today = False
        
        # 创建界面
        self.create_widgets()
        
        # 检查签到状态
        self.check_checkin_status()
          # 生成每日任务
        self.generate_daily_task()

    def center_window(self):
        """使窗口居中"""
        self.root.update_idletasks()
        width = 700
        height = 550
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_styles(self):
        """设置ttk样式"""
        style = ttk.Style()
        
        # 设置主题
        style.theme_use('clam')
        
        # 配置按钮样式
        style.configure('Custom.TButton',
                       font=('Microsoft YaHei', 10, 'bold'),
                       borderwidth=2,
                       relief='raised',
                       background='#4CAF50',
                       foreground='white')
        
        style.map('Custom.TButton',
                 background=[('active', '#45a049'),
                           ('pressed', '#3e8e41')])
        
        # 配置签到按钮样式
        style.configure('Checkin.TButton',
                       font=('Microsoft YaHei', 12, 'bold'),
                       borderwidth=3,
                       relief='raised',
                       background='#FF6B6B',
                       foreground='white')
        
        style.map('Checkin.TButton',
                 background=[('active', '#FF5252'),
                           ('pressed', '#E53935')])
        
        # 配置已签到按钮样式
        style.configure('Checked.TButton',
                       font=('Microsoft YaHei', 12, 'bold'),
                       borderwidth=3,
                       relief='sunken',
                       background='#9E9E9E',
                       foreground='white')
        
        # 配置标签样式
        style.configure('Title.TLabel',
                       font=('Microsoft YaHei', 16, 'bold'),
                       background='#f0f8ff',
                       foreground='#2E3A59')
        
        style.configure('Subtitle.TLabel',
                       font=('Microsoft YaHei', 12, 'bold'),
                       background='#f0f8ff',
                       foreground='#4A5568')
        
        style.configure('Info.TLabel',
                       font=('Microsoft YaHei', 10),
                       background='#f0f8ff',
                       foreground='#2D3748')


    def get_default_tasks(self):
        return {
            5: [
                "和宠物打个招呼吧"  # 手掌 打招呼
            ],
            12: [
                "左转，舒活舒活筋骨"  # 左转
            ],
            13: [
                "右转，舒活舒活筋骨"  # 右转
            ],
            2: [
                "大拇指，前进！"  # 前进
            ],
            3: [
                "和宠物聊聊天吧"
            ],
            4: [
                "嘘！让宠物趴下"  # 嘘 趴下
            ],
            11: [
                "OK，让宠物摇摇尾巴"  # ok 摇尾巴
            ],
            14: [
                "666，让宠物摇摆起来吧"  # 666 摇摆
            ]
        }

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架 - 减少边距
        main_frame = tk.Frame(self.root, bg='#f0f8ff')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # 创建顶部欢迎区域 - 减少间距
        welcome_frame = tk.Frame(main_frame, bg='#f0f8ff')
        welcome_frame.pack(fill='x', pady=(0, 10))
        
        # 顶部区域 - 宠物图片和标题 - 减少间距
        top_frame = tk.Frame(main_frame, bg='#f0f8ff')
        top_frame.pack(fill='x', pady=(5, 15))
        
        # 左侧宠物图片
        pet_frame = tk.Frame(top_frame, bg='#f0f8ff')
        pet_frame.pack(side='left', anchor='nw')
        
        # 加载宠物图片
        self.load_pet_image(pet_frame)
        
        # 右侧标题和日期
        info_frame = tk.Frame(top_frame, bg='#f0f8ff')
        info_frame.pack(side='right', fill='x', expand=True, padx=(15, 0))
        
        # 标题 - 调整到与欢迎标题等高 - 减小字体
        title_label = tk.Label(info_frame, text="🐱 AI智能宠物", 
                              font=('Microsoft YaHei', 20, 'bold'),
                              bg='#f0f8ff', fg='#2E3A59')
        title_label.pack(anchor='ne', pady=(0, 3))
        
        # 当前日期
        current_date = datetime.now().strftime("%Y年%m月%d日 %A")
        self.date_label = ttk.Label(info_frame, text=current_date, style='Info.TLabel')
        self.date_label.pack(anchor='ne', pady=(0, 3))        # 创建内容区域
        content_frame = tk.Frame(main_frame, bg='#f0f8ff')
        content_frame.pack(fill='both', expand=True)
        
        # 左侧区域 - 好感度和状态 - 固定宽度刚好容纳进度条
        left_frame = tk.LabelFrame(content_frame, text="宠物状态", font=('Microsoft YaHei', 11, 'bold'),
                                  bg='#ffffff', fg='#2E3A59', relief='ridge', borderwidth=2, width=220)
        left_frame.pack(side='left', fill='y', padx=(0, 5))
        left_frame.pack_propagate(False)  # 禁用自动调整大小
        
        # 好感度显示 - 减少间距
        affection_frame = tk.Frame(left_frame, bg='#ffffff')
        affection_frame.pack(fill='x', padx=15, pady=12)
        
        affection_title = ttk.Label(affection_frame, text="💖 好感度", style='Subtitle.TLabel')
        affection_title.config(background='#ffffff')
        affection_title.pack(anchor='w', pady=(0, 3))
        
        # 好感度进度条
        self.affection_var = tk.StringVar(value=f"{self.affection}/100")
        self.affection_progress = ttk.Progressbar(affection_frame, 
                                                 variable=self.affection_var,
                                                 maximum=100,
                                                 length=180,
                                                 mode='determinate',
                                                 style='TProgressbar')
        self.affection_progress.pack(anchor='w', pady=(0, 3))
        self.affection_progress['value'] = self.affection
        
        self.affection_label = ttk.Label(affection_frame, text=f"{self.affection}/100", 
                                        style='Info.TLabel')
        self.affection_label.config(background='#ffffff')
        self.affection_label.pack(anchor='w')
          # 宠物状态显示 - 减少间距
        status_frame = tk.Frame(left_frame, bg='#ffffff')
        status_frame.pack(fill='x', padx=15, pady=12)
        
        status_title = ttk.Label(status_frame, text="🎭 当前好感度状态", style='Subtitle.TLabel')
        status_title.config(background='#ffffff')
        status_title.pack(anchor='w', pady=(0, 3))
        
        self.status_label = ttk.Label(status_frame, text="😊 开心", style='Info.TLabel')
        self.status_label.config(background='#ffffff')
        self.status_label.pack(anchor='w')
          # 中间区域 - 任务和签到 - 更紧凑
        middle_frame = tk.LabelFrame(content_frame, text="每日活动", font=('Microsoft YaHei', 11, 'bold'),
                       bg='#ffffff', fg='#2E3A59', relief='ridge', borderwidth=2)
        middle_frame.pack(side='left', fill='both', expand=True, padx=(5, 5))

        # 右侧区域 - 3cats.jpg图片
        right_frame = tk.Frame(content_frame, bg='#f0f8ff')
        right_frame.pack(side='right', fill='both', expand=False, padx=(5, 0))

        # 加载3cats.jpg图片，保持完整宽度
        self.load_middle_image(right_frame)

        # 任务区域 - 更紧凑
        task_frame = tk.Frame(middle_frame, bg='#ffffff')
        task_frame.pack(fill='both', expand=True, padx=12, pady=10)

        task_title = ttk.Label(task_frame, text="📝 每日任务", style='Subtitle.TLabel')
        task_title.config(background='#ffffff')
        task_title.pack(anchor='w', pady=(0, 6))

        # 任务显示区域 - 居中对齐
        task_display_frame = tk.Frame(task_frame, bg='#f8f9fa', relief='sunken', borderwidth=1)
        task_display_frame.pack(fill='x', pady=(0, 6))

        self.task_label = tk.Label(task_display_frame, text="正在生成今日任务...", 
                      font=('Microsoft YaHei', 10),
                      bg='#f8f9fa', fg='#2D3748',
                      wraplength=180, justify='center')  # 改为居中对齐
        self.task_label.pack(padx=10, pady=10)

        # 任务状态 - 居中对齐
        self.task_status_label = tk.Label(task_frame, text="", 
                         font=('Microsoft YaHei', 9),
                         bg='#ffffff', fg='#666')
        self.task_status_label.pack(pady=(0, 6))  # 去掉anchor='w'，默认居中

        # 任务按钮 - 居中对齐
        task_button_frame = tk.Frame(task_frame, bg='#ffffff')
        task_button_frame.pack(fill='x')
        self.refresh_task_button = ttk.Button(task_button_frame, text="🔄 刷新任务", 
                             command=self.refresh_task,
                             style='Custom.TButton')
        self.refresh_task_button.pack(pady=(0, 6))  # 去掉side='left'，默认居中

        # 签到区域 - 更紧凑
        checkin_frame = tk.Frame(middle_frame, bg='#ffffff')
        checkin_frame.pack(fill='x', padx=12, pady=10)

        checkin_title = ttk.Label(checkin_frame, text="📅 每日签到", style='Subtitle.TLabel')
        checkin_title.config(background='#ffffff')
        checkin_title.pack(pady=(0, 6))  # 去掉anchor='w'，默认居中

        # 添加签到按钮 - 居中对齐
        self.checkin_button = ttk.Button(checkin_frame, text="🎯 今日签到", 
                                        command=self.daily_checkin,
                                        style='Checkin.TButton')
        self.checkin_button.pack(pady=(0, 6))  # 去掉anchor='w'，默认居中
        
        # 更新显示
        self.update_affection_display()
        self.update_status_display()

    def load_pet_image(self, parent_frame):
        """加载宠物图片"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(script_dir, 'cat.jpeg')
            
            if os.path.exists(image_path):
                # 加载并调整图片大小 - 减小图片尺寸
                pil_image = Image.open(image_path)
                pil_image = pil_image.resize((100, 100), Image.Resampling.LANCZOS)
                
                # 创建圆形遮罩
                mask = Image.new('L', (100, 100), 0)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, 100, 100), fill=255)
                
                # 应用遮罩
                output = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
                output.paste(pil_image, (0, 0))
                output.putalpha(mask)
                
                # 转换为tkinter可用格式
                self.pet_image = ImageTk.PhotoImage(output)
                
                # 直接显示图片，不加边框
                image_label = tk.Label(parent_frame, image=self.pet_image, bg='#f0f8ff')
                image_label.pack(anchor='nw')
        
                # 添加宠物名称
                name_label = tk.Label(parent_frame, text="🐾 胡萝卜", 
                                     font=('Microsoft YaHei', 10, 'bold'),
                                     bg='#f0f8ff', fg='#2E3A59')
                name_label.pack(anchor='w', pady=(5, 0))
        except Exception as e:
            print(f"加载宠物图片时出错: {e}")
            # 显示默认图标
            placeholder_frame = tk.Frame(parent_frame, bg='#f0f8ff', 
                                       width=100, height=100)
            placeholder_frame.pack(anchor='nw')
            placeholder_frame.pack_propagate(False)
            
            placeholder_label = tk.Label(placeholder_frame, text="🐱", 
                                       font=('Arial', 40),
                                       bg='#f0f8ff', fg='#666')
            placeholder_label.place(relx=0.5, rely=0.5, anchor='center')
              # 添加宠物名称
            name_label = tk.Label(parent_frame, text="🐾 胡萝卜", 
                                 font=('Microsoft YaHei', 10, 'bold'),
                                 bg='#f0f8ff', fg='#2E3A59')
            name_label.pack(anchor='w', pady=(5, 0))

    def load_middle_image(self, parent_frame):
        """加载右侧的3cats.jpg图片，保持完整宽度"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, '3cats.jpg')
        
        if os.path.exists(image_path):
            # 加载图片并保持原始宽高比
            pil_image = Image.open(image_path)
            
            # 获取原始尺寸
            original_width, original_height = pil_image.size
            
            # 设置目标高度为300像素，按比例调整宽度
            target_height = 300
            target_width = int(original_width * target_height / original_height)
            
            # 调整图片大小，保持原始宽高比
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 转换为tkinter可用格式
            self.middle_image = ImageTk.PhotoImage(pil_image)
            
            # 显示图片，填充整个右侧区域
            image_label = tk.Label(parent_frame, image=self.middle_image, bg='#f0f8ff')
            image_label.pack(fill='both', expand=True)
            
        else:
            print(f"3cats.jpg文件不存在: {image_path}")
            # 显示占位符
            placeholder_label = tk.Label(parent_frame, text="🐱🐱🐱", 
                                        font=('Arial', 24),
                                        bg='#f0f8ff', fg='#666')
            placeholder_label.pack(pady=10)
            
    
    def update_affection_display(self):
        """更新好感度显示"""
        self.affection_label.config(text=f"{self.affection}/100")
        self.affection_progress['value'] = self.affection
        
        # 更新进度条颜色
        if self.affection >= 80:
            style = ttk.Style()
            style.configure("TProgressbar", background='#4CAF50')
        elif self.affection >= 50:
            style = ttk.Style()
            style.configure("TProgressbar", background='#FFC107')
        else:
            style = ttk.Style()
            style.configure("TProgressbar", background='#F44336')

    def update_status_display(self):
        """更新宠物状态显示"""
        if self.affection >= 80:
            status = "😍 非常开心"
        elif self.affection >= 60:
            status = "😊 开心"
        elif self.affection >= 40:
            status = "😐 普通"
        elif self.affection >= 20:
            status = "😔 有点沮丧"
        else:
            status = "😢 很沮丧"
        
        self.status_label.config(text=status)

    def check_checkin_status(self):
        """检查签到状态"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.last_checkin == today:
            self.checkin_button.config(text="✅ 今日已签到", state='disabled', 
                                     style='Checked.TButton')
        else:
            self.checkin_button.config(text="🎯 今日签到", state='normal', 
                                     style='Checkin.TButton')

    def daily_checkin(self):
        """每日签到"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.last_checkin == today:
            messagebox.showinfo("签到提醒", "今天已经签到过了哦！明天再来吧~")
            return
        
        # 执行签到
        self.last_checkin = today
        affection_gain = random.randint(3, 8)
        self.affection = min(100, self.affection + affection_gain)
        
        # 更新UI
        self.update_affection_display()
        self.update_status_display()
        self.check_checkin_status()
        
        # 签到成功消息
        messages = [
            f"签到成功！好感度+{affection_gain}💕",
            f"今日签到完成！获得{affection_gain}点好感度~",
            f"签到奖励：好感度+{affection_gain}，继续加油！",
            f"每日签到√ 好感度提升{affection_gain}点！"
        ]
        
        messagebox.showinfo("签到成功", random.choice(messages))

    def generate_daily_task(self):
        """生成每日任务"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 如果今天已经生成过任务且任务已完成，则不重新生成
        if self.last_task_date == today and self.task_completed_today:
            return
        
        # 如果是新的一天，重置任务状态
        if self.last_task_date != today:
            self.task_completed_today = False
            self.last_task_date = today
        
        # 直接使用get_default_tasks，随机选择任务类型
        task_type = random.choice(list(self.task_dict.keys()))
        
        # 从该类型中随机选择一个任务
        task_list = self.task_dict[task_type]
        task_description = random.choice(task_list)
        
        # 设置当前任务
        self.current_task = {
            'type': task_type,
            'description': task_description,
            'completed': False
        }
        
        # 更新任务显示
        self.update_task_display()

    def update_task_display(self):
        """更新任务显示"""
        if self.current_task:
            task_text = f"📋 今日任务\n{self.current_task['description']}"
            self.task_label.config(text=task_text)
            
            if self.task_completed_today:
                self.task_status_label.config(text="✅ 任务已完成", fg='#4CAF50')
            else:
                self.task_status_label.config(text="⏳ 等待完成", fg='#FF9800')
        else:
            self.task_label.config(text="正在生成今日任务...")
            self.task_status_label.config(text="")

    def refresh_task(self):
        """刷新任务"""
        # 检查是否可以刷新任务
        if self.task_completed_today:
            messagebox.showinfo("任务提醒", "今日任务已完成，明天再来刷新新任务吧！")
            return
        
        # 直接使用get_default_tasks，获取当前任务类型，避免重复
        current_task_type = self.current_task['type'] if self.current_task else None
        available_types = [t for t in self.task_dict.keys() if t != current_task_type]
        
        if not available_types:
            available_types = list(self.task_dict.keys())
        
        # 随机选择新的任务类型
        task_type = random.choice(available_types)
        
        # 获取当前任务描述，避免重复
        current_description = self.current_task['description'] if self.current_task else None
        task_list = self.task_dict[task_type]
        available_tasks = [t for t in task_list if t != current_description]
        
        if not available_tasks:
            available_tasks = task_list
        
        # 随机选择新任务
        task_description = random.choice(available_tasks)
        
        # 设置新任务
        self.current_task = {
            'type': task_type,
            'description': task_description,
            'completed': False
        }
        
        # 更新任务显示
        self.update_task_display()
        
        # 显示刷新成功消息
        refresh_messages = [
            "任务已刷新！准备好接受新挑战了吗？",
            "新任务来了！让我们一起努力完成吧~",
            "任务更新成功！这次会更有趣哦！",
            "刷新完成！新的任务等着你呢！"
        ]
        
        messagebox.showinfo("任务刷新", random.choice(refresh_messages))

    def mark_task_completed(self):
        """标记任务完成（供外部调用）"""
        if self.current_task and not self.task_completed_today:
            self.task_completed_today = True
            self.current_task['completed'] = True
            
            # 增加好感度
            affection_gain = random.randint(5, 12)
            self.affection = min(100, self.affection + affection_gain)
            
            # 更新显示
            self.update_task_display()
            self.update_affection_display()
            self.update_status_display()
            
            return True
        return False

    def get_current_task(self):
        """获取当前任务信息（供外部调用）"""
        return self.current_task

    def run(self):
        """运行主循环"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = PetUI(initial_affection=50)
        app.run()
    except Exception as e:
        print(f"运行AI宠物UI时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
