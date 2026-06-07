import random

# ==================== 1. 六都精細景點資料庫 ====================
# "not_allowed": 該景點不適合的交通工具類型
travel_data = {
    "台北市": {
        "sunny": [
            {"name": "台北101", "not_allowed": []},
            {"name": "象山步道", "not_allowed": ["bicycle"]},    # 陡坡樓梯，單車不宜
            {"name": "陽明山國家公園", "not_allowed": ["bicycle"]}, # 山路長且陡，休閒單車不建議
            {"name": "西門町徒步區", "not_allowed": ["car"]},       # 汽車無法進入、極難停車
            {"name": "大稻埕碼頭", "not_allowed": []}
        ],
        "rainy": [
            {"name": "國立故宮博物院", "not_allowed": []},
            {"name": "華山1914文創園區", "not_allowed": []},
            {"name": "松山文創園區", "not_allowed": []},
            {"name": "台北地下街", "not_allowed": ["car"]}       # 周邊大塞車且都在地下
        ]
    },
    "新北市": {
        "sunny": [
            {"name": "九份老街", "not_allowed": ["car", "bicycle"]}, # 狹窄山路階梯，大車易塞死
            {"name": "十分瀑布放天燈", "not_allowed": []},
            {"name": "淡水老街", "not_allowed": ["car"]},         # 人潮擁擠，汽車無法進老街
            {"name": "野柳地質公園", "not_allowed": ["bicycle"]},   # 需走濱海公路，單車較危險
            {"name": "碧潭風景區", "not_allowed": []}
        ],
        "rainy": [
            {"name": "十三行博物館", "not_allowed": []},
            {"name": "淡水紅毛城", "not_allowed": []},
            {"name": "板橋林本源園邸", "not_allowed": []},
            {"name": "鶯歌陶瓷博物館", "not_allowed": []}
        ]
    },
    "桃園市": {
        "sunny": [
            {"name": "大溪老街", "not_allowed": ["car"]},         # 假日管制徒步區，開車極難停
            {"name": "石門水庫風景區", "not_allowed": []},
            {"name": "拉拉山巨木群", "not_allowed": ["bicycle"]},   # 北橫山路極長且陡，考驗腳力
            {"name": "永安漁港海祭廣場", "not_allowed": []},
            {"name": "虎頭山創新園區", "not_allowed": []}
        ],
        "rainy": [
            {"name": "Xpark 水族館", "not_allowed": []},
            {"name": "桃園市圖書館總館", "not_allowed": []},
            {"name": "大溪老茶廠", "not_allowed": []},
            {"name": "手信霧隱城", "not_allowed": []}
        ]
    },
    "台中市": {
        "sunny": [
            {"name": "高美濕地木棧道", "not_allowed": []},
            {"name": "彩虹眷村", "not_allowed": []},
            {"name": "東海大學路思義教堂", "not_allowed": ["car"]}, # 校內進行汽車管制
            {"name": "草悟道 / 市民廣場", "not_allowed": []},
            {"name": "后豐鐵馬道", "not_allowed": ["car"]}        # 專屬自行車道，汽車禁止進入
        ],
        "rainy": [
            {"name": "宮原眼科", "not_allowed": ["car"]},         # 火車站旁市中心，極難停汽車
            {"name": "國立自然科學博物館", "not_allowed": []},
            {"name": "審計新村文創市集", "not_allowed": ["car"]},   # 周邊巷弄狹窄難停車
            {"name": "台中三井 Outlet", "not_allowed": []}
        ]
    },
    "台南市": {
        "sunny": [
            {"name": "安平古堡 & 樹屋", "not_allowed": []},
            {"name": "奇美博物館", "not_allowed": []},
            {"name": "神農街老屋漫步", "not_allowed": ["car"]},    # 極窄古老巷弄，汽車進不去
            {"name": "七股鹽山", "not_allowed": []},
            {"name": "水晶教堂 / 北門鹽田", "not_allowed": ["bicycle"]} # 靠海路途遙遠，不建議休閒單車
        ],
        "rainy": [
            {"name": "台南市美術館二館", "not_allowed": []},
            {"name": "國立臺灣文學館", "not_allowed": []},
            {"name": "林百貨", "not_allowed": ["car"]},           # 圓環鬧區核心，非常難停汽車
            {"name": "國立臺灣歷史博物館", "not_allowed": []}
        ]
    },
    "高雄市": {
        "sunny": [
            {"name": "駁二藝術特區", "not_allowed": []},
            {"name": "旗津老街 & 燈塔", "not_allowed": ["car"]},  # 需搭渡輪或走過港隧道，老街管制汽車
            {"name": "蓮池潭龍虎塔", "not_allowed": []},
            {"name": "西子灣看夕陽", "not_allowed": []},
            {"name": "壽山動物園", "not_allowed": ["bicycle"]}     # 上山山路較陡且有野生獼猴搶食
        ],
        "rainy": [
            {"name": "高雄國立科學工藝博物館", "not_allowed": []},
            {"name": "衛武營國家藝術文化中心", "not_allowed": []},
            {"name": "高雄市立美術館", "not_allowed": []},
            {"name": "SKM Park Outlets 大魯閣", "not_allowed": []}
        ]
    }
}

# 確保相容性：若未來有人直接輸入景點字串，自動補齊字典結構
for city_name, content in travel_data.items():
    for weather_type in ["sunny", "rainy"]:
        for i, place in enumerate(content[weather_type]):
            if isinstance(place, str):
                content[weather_type][i] = {"name": place, "not_allowed": []}


# ==================== 2. 六都美食與在地交通加值資料 ====================
extra_info = {
    "台北市": {"food": "牛肉麵、小籠包、大稻埕迪化街中藥乾貨", "transport": "捷運 (MRT)、公車、YouBike"},
    "新北市": {"food": "淡水阿給、九份芋圓、深坑臭豆腐", "transport": "捷運、輕軌、公車、客運"},
    "桃園市": {"food": "大溪豆干、龍岡米干、石門活魚三吃", "transport": "開車、桃園機場捷運、公車"},
    "台中市": {"food": "太陽餅、逢甲夜市小吃、屋馬燒肉", "transport": "捷運、公車、iBike"},
    "台南市": {"food": "牛肉湯、鱔魚意麵、小卷米粉、生魚片", "transport": "租機車、慢活步行、公車"},
    "高雄市": {"food": "旗津海鮮、丹丹漢堡、鴨肉珍、汕頭火鍋", "transport": "高雄捷運、環狀輕軌、YouBike"}
}

vehicle_map = {
    "car": "🚗 汽車",
    "motorcycle": "🛵 機車",
    "bicycle": "🚲 腳踏車"
}

# ==================== 3. 核心行程規劃演算法 ====================
def generate_itinerary(city, days, weather, vehicle):
    if city not in travel_data:
        print(f"❌ 找不到【{city}】的資料。請確認是否輸入完整的六都名稱（如：台北市）。")
        return

    # 1. 根據天氣撈取初步景點
    if weather == "y":
        all_places = travel_data[city]["rainy"]
        print("\n☔ 天氣預報：雨天（已為您啟動舒適的室內備案行程）")
    else:
        all_places = travel_data[city]["sunny"]
        print("\n☀️ 天氣預報：晴天（已為您啟動陽光戶外休閒行程）")

    # 2. 根據交通工具篩選符合的景點
    filtered_places = []
    skipped_places = []
    
    for place in all_places:
        if vehicle in place["not_allowed"]:
            skipped_places.append(place["name"])
        else:
            filtered_places.append(place["name"])

    # 如果該交通工具把景點都過濾光了
    if not filtered_places:
        print(f"⚠️ 糟糕！{city}目前沒有適合【{vehicle_map[vehicle]}】的景點，請考慮更換交通工具。")
        return

    # 亂數打亂行程
    random.shuffle(filtered_places)

    print("\n" + "="*40)
    print(f"🗺️  {city} {days} 天旅遊推薦 ({vehicle_map[vehicle]}自駕版)")
    print("="*40 + "\n")

    # 3. 智慧型分配演算法（使用餘數分配法，確保天數多時景點能均勻拆分，不會集中在第一天）
    days_itinerary = [[] for _ in range(days)]
    for i, place in enumerate(filtered_places):
        days_itinerary[i % days].append(place)

    # 4. 輸出每日行程
    for d in range(days):
        print(f"📅 第 {d+1} 天")
        if days_itinerary[d]:
            for p in days_itinerary[d]:
                print(f"  📍 {p}")
        else:
            print("  （此日無合適推薦景點，建議安排飯店設施或市區漫步）")
        print()

    # 提示因為車型被過濾掉的遺珠景點
    if skipped_places:
        print(f"💡 註：以下景點因路寬限制、山路過陡或不易停放【{vehicle_map[vehicle]}】，系統已自動避開：")
        print(f"   ❌ {', '.join(skipped_places)}\n")


def enhanced_service(city, weather):
    print("-" * 40)
    if city in extra_info:
        print(f"💡 【{city}】在地旅遊補給站：")
        print(f"🍴 當地名產：{extra_info[city]['food']}")
        print(f"🚲 推薦交通：{extra_info[city]['transport']}")
        
        # 天氣感應美食邏輯
        print("🍲 天氣推薦療癒美食：", end="")
        if weather == "y":
            print("【今天下雨/體感較涼】推薦喝碗暖呼呼的在地熱湯，暖胃又除濕！")
        else:
            print("【今天晴空萬里/陽光普照】天氣熱呼呼！強力推薦來碗在地刨冰或手搖冷飲消消暑！")
    else:
        print(f"💡 溫馨提醒：造訪 {city} 時，別忘了體驗當地獨特的風土民情喔！")
    print("-" * 40)


# ==================== 4. 主程式執行入口 ====================
if __name__ == "__main__":
    print("👉 系統目前已建置【台灣六都】精細路線規劃：")
    available_cities = list(travel_data.keys())
    for i, c in enumerate(available_cities, 1):
        print(c, end="  ")
    print("\n")

    # 使用者輸入與名稱模糊修正防呆
    raw_city = input("請輸入縣市（例如：台北市、台南市）：").strip()
    city = raw_city
    if not city.endswith(("市", "縣")):
        for candidate in available_cities:
            if city in candidate:
                city = candidate
                break

    # 旅遊天數防呆
    try:
        days = int(input("請輸入旅遊天數："))
        if days <= 0: days = 1
    except ValueError:
        print("⚠️ 天數輸入錯誤，預設為 1 天")
        days = 1

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
    feedback = input("您對這次的 AI 推薦滿意嗎？(y/n): ").strip().lower()
    if feedback == 'y':
        print("✨ 祝您旅途愉快！路上請注意安全。")
    else:
        print("✉️ 感謝回饋，我們會持續優化景點路線限制與美食資料庫。")

