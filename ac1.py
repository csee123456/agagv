import time
import random
import threading

# --- 1. AGV 模擬模組 ---
class AGV_Simulator(threading.Thread):
    def __init__(self, agv_id, x=0, y=0):
        super().__init__()
        self.agv_id = agv_id
        self.x = x
        self.y = y
        self.battery = 100.0
        self.status = "Idle"
        self.daemon = True

    def execute_task(self, task):
        self.status = "Moving"
        dist = abs(self.x - task['x']) + abs(self.y - task['y'])
        
        # 顯示任務類型，讓輸出更清楚
        tag = "🚨 [緊急]" if task['type'] == "緊急" else "📦 [一般]"
        print(f"{tag} {self.agv_id} 開始處理 {task['id']} (權重: {task['priority']})")
        
        time.sleep(dist * 0.2) 
        
        self.battery -= dist * 0.5
        self.x, self.y = task['x'], task['y']
        self.status = "Idle"
        print(f"✅ {self.agv_id} 完成任務 {task['id']}，目前電量: {self.battery:.1f}%")

# --- 2. 任務產生模組 (保留原本的緊急判斷) ---
class Task_Generator:
    def __init__(self):
        self.task_count = 0

    def generate(self):
        self.task_count += 1
        priority = random.randint(1, 10) # 1~10 權重
        return {
            "id": f"TASK_{self.task_count:03d}",
            "x": random.randint(0, 20),
            "y": random.randint(0, 20),
            "priority": priority,
            "type": "緊急" if priority > 7 else "一般" # 權重 > 7 為緊急
        }

# --- 3. 主控制流程 ---
def main():
    print("🚀 優先權調度系統啟動")
    
    agv_list = [AGV_Simulator(f"AGV_{i+1:02d}", random.randint(0,5), random.randint(0,5)) for i in range(3)]
    for agv in agv_list: agv.start()
        
    generator = Task_Generator()
    task_queue = [] # 用來存放尚未被分配的任務池

    # 模擬產生 10 個任務並放入池中
    for _ in range(10):
        task_queue.append(generator.generate())

    # 【核心修改：優先權排序】
    # 根據 priority 從大到小排序，確保緊急的（權重高）排在前面
    task_queue.sort(key=lambda x: x['priority'], reverse=True)

    print(f"📋 已產生 10 個任務，優先順序：{[t['id']+'('+str(t['priority'])+')' for t in task_queue]}")

    # 開始分配任務
    while task_queue:
        current_task = task_queue[0] # 取出目前最高優先權的任務
        
        assigned = False
        for agv in agv_list:
            if agv.status == "Idle":
                # 指派任務
                t = threading.Thread(target=agv.execute_task, args=(current_task,))
                t.start()
                task_queue.pop(0) # 從待辦清單移除
                assigned = True
                break
        
        if not assigned:
            time.sleep(0.5) # 若所有 AGV 忙碌，等待後再試

    # 等待所有動作結束
    time.sleep(10)
    print("\n=== 所有優先任務處理完畢 ===")

if __name__ == "__main__":
    main()