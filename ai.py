#AI
# ===== 台灣旅遊行程推薦系統（全台縣市＋雨天備案＋車型路線限制＋天氣美食）=====

import random

# 景點資料庫：加入交通工具限制標籤
# "not_allowed": 該景點不適合的交通工具類型
travel_data = {
    "台北市": {
        "sunny": [
            {"name": "台北101", "not_allowed": []},
            {"name": "象山步道", "not_allowed": ["bicycle"]},  # 陡坡、樓梯，單車不宜
            {"name": "陽明山", "not_allowed": ["bicycle"]},   # 山路長且陡，休閒單車不建議
            {"name": "西門町", "not_allowed": ["car"]},       # 徒步區、極難停車
            {"name": "大稻埕碼頭", "not_allowed": []}
        ],
        "rainy": [
            {"name": "故宮博物院", "not_allowed": []},
            {"name": "華山文創園區", "not_allowed": []},
            {"name": "松山文創園區", "not_allowed": []},
            {"name": "台北地下街", "not_allowed": ["car"]}     # 周邊塞車且都在地下
        ]
    },
    "新北市": {
        "sunny": [
            {"name": "九份老街", "not_allowed": ["car", "bicycle"]}, # 狹窄山路、階梯、大車易塞死
            {"name": "十分瀑布", "not_allowed": []},
            {"name": "淡水老街", "not_allowed": ["car"]}, # 人潮擁擠、汽車無法進入老街內
            {"name": "野柳地質公園", "not_allowed": ["bicycle"]}
        ],
        "rainy": [
            {"name": "十三行博物館", "not_allowed": []},
            {"name": "淡水紅毛城", "not_allowed": []},
            {"name": "林本源園邸", "not_allowed": []}
        ]
    },
    "台中市": {
        "sunny": [
            {"name": "高美濕地", "not_allowed": []},
            {"name": "彩虹眷村", "not_allowed": []},
            {"name": "東海大學", "not_allowed": ["car"]}, # 校內汽車管制
            {"name": "草悟道", "not_allowed": []}
        ],
        "rainy": [
            {"name": "宮原眼科", "not_allowed": ["car"]}, # 火車站旁極難停汽車
            {"name": "國立自然科學博物館", "not_allowed": []},
            {"name": "審計新村", "not_allowed": ["car"]},
            {"name": "三井Outlet", "not_allowed": []}
        ]
    }
    # ... 其他縣市可以依此類推加入 "not_allowed" 標籤
}

# 補齊其他未特別設定標籤的縣市預設結構
for city_name, content in travel_data.items():
    for weather_type in ["sunny", "rainy"]:
        for i, place in enumerate(content[weather_type]):
            if isinstance(place, str):
                content[weather_type][i] = {"name": place, "not_allowed": []}


# 擴充功能：美食與交通加值服務
extra_info = {
    "台北市": {"food": "牛肉麵、小籠包", "transport": "捷運、公車"},
    "新北市": {"food": "阿給、九份芋圓", "transport": "捷運、公車、客運"},
    "基隆市": {"food": "沙拉船、鼎邊趖", "transport": "火車、客運"},
    "桃園市": {"food": "大溪豆干", "transport": "開車、客運、捷運"},
    "新竹市": {"food": "新竹貢丸、米粉", "transport": "開車、客運"},
    "新竹縣": {"food": "客家菜包、粄條", "transport": "開車、客運"},
    "苗栗縣": {"food": "客家小炒", "transport": "開車、客運"},
    "台中市": {"food": "太陽餅、逢甲夜市小吃", "transport": "公車、iBike"},
    "彰化縣": {"food": "肉圓、爌肉飯", "transport": "開車、客運"},
    "南投縣": {"food": "南投意麵、茶葉蛋", "transport": "開車、客運"},
    "雲林縣": {"food": "古坑咖啡、鴨肉麵線", "transport": "開車、客運"},
    "嘉義市": {"food": "火雞肉飯", "transport": "開車、客運"},
    "嘉義縣": {"food": "奮起湖便當", "transport": "開車、客運"},
    "台南市": {"food": "牛肉湯、鱔魚意麵", "transport": "機車、慢活步行"},
    "高雄市": {"food": "海鮮、丹丹漢堡", "transport": "輕軌、捷運"},
    "屏東縣": {"food": "萬巒豬腳、黑鮪魚", "transport": "開車、客運"},
    "宜蘭縣": {"food": "三星蔥餅、鴨賞", "transport": "開車、客運"},
    "花蓮縣": {"food": "炸蛋蔥油餅、扁食", "transport": "開車、客運"},
    "台東縣": {"food": "池上便當、卑南豬血湯", "transport": "開車、客運"},
    "澎湖縣": {"food": "仙人掌冰、黑糖糕", "transport": "機車、開車"},
    "金門縣": {"food": "高粱酒、貢糖、廣東粥", "transport": "機車、開車"},
    "連江縣": {"food": "老酒麵線、魚麵", "transport": "機車"}
}

# 交通工具對照表
vehicle_map = {
    "car": "🚗 汽車",
    "motorcycle": "🛵 機車",
    "bicycle": "🚲 腳踏車"
}

def generate_itinerary(city, days, weather, vehicle):
    if city not in travel_data:
        print("❌ 沒有這個縣市的資料")
        return

    # 1. 根據天氣撈取初步景點
    if weather == "y":
        all_places = travel_data[city]["rainy"]
        print("☔ 天氣預報：雨天（啟動室內備案行程）")
    else:
        all_places = travel_data[city]["sunny"]
        print("☀️ 天氣預報：晴天（啟動戶外休閒行程）")

    # 2. 根據交通工具篩選「車子進得去/適合去」的景點
    filtered_places = []
    skipped_places = []
    
    for place in all_places:
        if vehicle in place["not_allowed"]:
            skipped_places.append(place["name"])
        else:
            filtered_places.append(place["name"])

    # 亂數打亂行程
    random.shuffle(filtered_places)

    # 計算每天規劃的景點數量
    if len(filtered_places) == 0:
        print(f"⚠️ 糟糕！該縣市目前沒有適合 {vehicle_map[vehicle]} 的景點，請考慮更換交通工具。")
        return
        
    per_day = max(1, len(filtered_places) // days)

    print("\n==========================")
    print(f"🗺️  {city} {days}天旅遊推薦 ({vehicle_map[vehicle]}自駕版)")
    print("==========================\n")

    for d in range(days):
        start = d * per_day
        # 如果是最後一天，就把剩下的景點全部塞進去
        end = start + per_day if d < days - 1 else len(filtered_places)
        
        day_places = filtered_places[start:end]
        
        print(f"📅 第 {d+1} 天")
        if day_places:
            for p in day_places:
                print(f"  📍 {p}")
        else:
            print("  （此日無合適推薦景點，建議市區自由漫步）")
        print()

    # 提示因為車型被過濾掉的遺珠景點
    if skipped_places:
        print(f"💡 註：以下景點因路寬限制、山路過陡或不易停放【{vehicle_map[vehicle]}】，系統已自動避開：")
        print(f"   ❌ {', '.join(skipped_places)}\n")


def enhanced_service(city, weather):
    print("--------------------------")
    if city in extra_info:
        print(f"💡 【{city}】在地旅遊資訊：")
        print(f"🍴 當地名產：{extra_info[city]['food']}")
        print(f"🚲 推薦交通：{extra_info[city]['transport']}")
        
        # 天氣感應美食邏輯
        print("🍲 天氣推薦療癒美食：", end="")
        if weather == "y":
            print("【今天下雨/體感較涼】推薦喝碗暖呼呼的薑茶、麻油雞或熱湯，暖胃又除濕！")
        else:
            print("【今天晴空萬里/陽光普照】天氣熱呼呼！強力推薦來碗在地刨冰或手搖冷飲消消暑！")
    else:
        print(f"💡 溫馨提醒：造訪 {city} 時，別忘了體驗當地獨特的風土民情喔！")
    print("--------------------------")


# ===== 主程式執行節點 =====

print("👉 以下縣市可以按照車子大小規劃路線：")
for i, c in enumerate(travel_data.keys(), 1):
    print(c, end="  ")
    if i % 6 == 0: print() # 每6個縣市換一行，方便閱讀
print("\n")

# 使用者輸入
city = input("請輸入縣市：").strip()
days = int(input("請輸入旅遊天數："))
weather = input("是否下雨 (y/n)：").strip().lower()

print("\n🚗 請選擇您本次旅遊的主要交通工具：")
print("1. 汽車 (car)")
print("2. 機車 (motorcycle)")
print("3. 腳踏車 (bicycle)")
vehicle_choice = input("請輸入編號或英文 (1/2/3 或 car/motorcycle/bicycle)：").strip().lower()

# 轉換交通工具輸入值
if vehicle_choice in ["1", "car"]:
    vehicle = "car"
elif vehicle_choice in ["2", "motorcycle"]:
    vehicle = "motorcycle"
else:
    vehicle = "bicycle"

# 產出行程與加值服務
generate_itinerary(city, days, weather, vehicle)
enhanced_service(city, weather)

# 系統回饋
print("\n[ 系統回饋 ]")
feedback = input("您對這次的 AI 推薦滿意嗎？(y/n): ")
if feedback.lower() == 'y':
    print("✨ 祝您旅途愉快！路上請注意安全。")
else:
    print("✉️ 感謝回饋，我們會持續優化景點路線限制與美食資料庫。")