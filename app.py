import streamlit as st
import mediapipe as mp
import streamlit as st

# ↓この1行を追加！画面に「どこから読み込んでいるか」を表示させます
st.write("MediaPipeの場所:", mp.__file__)
# ↓この行の "📷" の部分を、ファイル名に書き換えます
st.set_page_config(page_title="顔バランス＆肌比較診断アプリ", page_icon="my_icon.png")
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageOps
import math

# --- 設定と関数定義 ---
st.set_page_config(page_title="顔バランス＆肌比較診断", layout="wide")

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# サイドバー設定
st.sidebar.title("設定")
edge_threshold = st.sidebar.slider("しわ検出感度", 50, 250, 150, help("数値を下げると細かい線を拾い、上げると深い線だけ拾います。"))
st.sidebar.info("※感度は両方の画像に同じように適用されます。")

def load_and_fix_image(uploaded_file):
    """画像を読み込み、回転を直し、リサイズする関数"""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    
    max_width = 600
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height))
    
    return np.array(image)

def calculate_distance(p1, p2):
    """2点間の距離を計算する"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def draw_mesh_and_box(image, landmarks_proto):
    """
    網目（メッシュ）と外枠（ボックス）を両方描画し、比率を計算する
    """
    h, w, _ = image.shape
    annotated_image = image.copy()
    landmarks_list = landmarks_proto.landmark

    # 1. 網目（メッシュ）を描画
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_proto,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_proto,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # 2. 基準点の取得（縦横計測用）
    top_idx = 10     # おでこの上
    bottom_idx = 152 # あご先
    left_idx = 234   # 左耳横（頬骨）
    right_idx = 454  # 右耳横（頬骨）
    
    pt_top = (int(landmarks_list[top_idx].x * w), int(landmarks_list[top_idx].y * h))
    pt_bottom = (int(landmarks_list[bottom_idx].x * w), int(landmarks_list[bottom_idx].y * h))
    pt_left = (int(landmarks_list[left_idx].x * w), int(landmarks_list[left_idx].y * h))
    pt_right = (int(landmarks_list[right_idx].x * w), int(landmarks_list[right_idx].y * h))
    
    # 3. 縦横ライン（十字）を描く
    cv2.line(annotated_image, pt_top, pt_bottom, (0, 255, 255), 3) # 縦線（黄色・太め）
    cv2.line(annotated_image, pt_left, pt_right, (0, 255, 255), 3) # 横線（黄色・太め）

    # 4. 外枠（バウンディングボックス）を描く
    x_coords = [pt_top[0], pt_bottom[0], pt_left[0], pt_right[0]]
    y_coords = [pt_top[1], pt_bottom[1], pt_left[1], pt_right[1]]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # ボックス描画（青色・太め）
    cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)

    # 5. 比率計算
    vertical_dist = calculate_distance(pt_top, pt_bottom)
    horizontal_dist = calculate_distance(pt_left, pt_right)
    
    if horizontal_dist == 0: return annotated_image, 0
    ratio = vertical_dist / horizontal_dist

    return annotated_image, ratio

def analyze_area(image, landmarks_list, indices, area_name):
    """指定エリアを切り抜き、エッジ（しわ）スコアを計算する関数"""
    image_height, image_width, _ = image.shape
    pts = []
    for idx in indices:
        pt = landmarks_list[idx]
        pts.append([int(pt.x * image_width), int(pt.y * image_height)])
    
    if not pts: return 0, None

    pts = np.array(pts)
    x, y, w, h = cv2.boundingRect(pts)
    
    margin = 10
    x, y = max(0, x-margin), max(0, y-margin)
    w, h = min(w+margin*2, image_width-x), min(h+margin*2, image_height-y)
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0: return 0, None

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 1.5)
    score = np.sum(edges > 0) / edges.size * 1000
    
    return score, (x, y, w, h)

def process_image(img_array, face_mesh):
    """1枚の画像を処理して結果を返す関数"""
    results = face_mesh.process(img_array)
    if not results.multi_face_landmarks:
        return None

    face_landmarks_proto = results.multi_face_landmarks[0] 
    face_landmarks_list = face_landmarks_proto.landmark

    mesh_box_img, ratio = draw_mesh_and_box(img_array, face_landmarks_proto)
    
    analyzed_img = img_array.copy()
    scores = {}

    forehead_idx = [109, 338, 9, 336, 151]
    nasolabial_idx = [
        205, 203, 36, 101, 50, 123, 117, 111, 147, 187, 207, 
        425, 423, 266, 330, 280, 352, 346, 340, 376, 411, 427
    ]
    marionette_idx = [
        57, 186, 212, 287, 410, 432, 273, 335, 406, 313, 18, 83, 182, 106, 43
    ]

    areas = {
        "おでこ": (forehead_idx, (0, 255, 0)),
        "ほうれい線周辺": (nasolabial_idx, (255, 165, 0)),
        "口元・あご周り": (marionette_idx, (255, 0, 255))
    }

    for name, (indices, color) in areas.items():
        score, rect = analyze_area(img_array, face_landmarks_list, indices, name)
        scores[name] = score
        if rect:
            x, y, w, h = rect
            cv2.rectangle(analyzed_img, (x, y), (x+w, y+h), color, 2)

    return mesh_box_img, analyzed_img, scores, ratio

# --- メイン画面構成 ---
# タイトル部分（アイコンと文字を横並びにする）
col1, col2 = st.columns([1, 6])  # 画面を1:6の比率で分けます

with col1:
    st.image("my_icon.png")      # 左側にアイコンを表示

with col2:
    st.title("顔バランス＆肌比較診断") # 右側にタイトルを表示
    
st.write("2枚の写真をアップロードして、骨格（網目・外枠）と肌の状態を比較します。")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("画像A (Before)", type=["jpg", "png"], key="a")
with col2:
    file_b = st.file_uploader("画像B (After)", type=["jpg", "png"], key="b")

if file_a and file_b:
    with st.spinner("2枚の画像を解析中..."):
        img_a = load_and_fix_image(file_a)
        img_b = load_and_fix_image(file_b)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

            result_a = process_image(img_a, face_mesh)
            result_b = process_image(img_b, face_mesh)

        if result_a and result_b:
            mesh_box_a, analyzed_a, scores_a, ratio_a = result_a
            mesh_box_b, analyzed_b, scores_b, ratio_b = result_b

            st.divider()
            st.header("1. 骨格・ゆがみ・比率")
            
            # 【追加】網目とゆがみの見方ガイド
            st.markdown("""
            **🤔 網目（メッシュ）でのゆがみのチェック方法**
            * **中心線（縦の水色線）**: 鼻筋からあごにかけて真っ直ぐですか？「く」の字に曲がっていませんか？
            * **左右のバランス**: 網目のマス目の大きさは左右同じですか？（片方だけ広いと、そちらが膨らんでいる可能性があります）
            * **高さの違い**: 目のラインや口角のライン（網目の横線）は水平ですか？
            """)

            col1_mesh, col2_mesh = st.columns(2)
            col1_mesh.image(mesh_box_a, caption=f"画像A: 縦横比 {ratio_a:.3f}", use_container_width=True)
            col2_mesh.image(mesh_box_b, caption=f"画像B: 縦横比 {ratio_b:.3f}", use_container_width=True)

            # 比率データの表示
            st.subheader("■ 顔の縦横比（Aspect Ratio）")
            # 【修正】文言を変更しました
            st.caption("※縦の長さ（おでこ〜あご） ÷ 横幅（頬骨の端〜端）で計算しています。")

            m_col1, m_col2 = st.columns(2)
            delta_ratio = ratio_b - ratio_a
            
            with m_col1:
                st.metric("画像A 比率", f"{ratio_a:.3f}")
                if ratio_a > 1.35:
                    st.info("ℹ️ **判定目安: 面長寄り**\n縦のラインが強調されています。")
                elif ratio_a < 1.25:
                    st.info("ℹ️ **判定目安: 丸顔・横幅寄り**\n丸みや横幅があるバランスです。")
                else:
                    st.info("ℹ️ **判定目安: 標準的な卵型バランス**")

            with m_col2:
                st.metric("画像B 比率", f"{ratio_b:.3f}", delta=f"{delta_ratio:.3f}", help="プラス＝縦長へ、マイナス＝丸顔・小顔へ変化")
                if ratio_b > 1.35:
                    st.info("ℹ️ **判定目安: 面長寄り**")
                elif ratio_b < 1.25:
                    st.info("ℹ️ **判定目安: 丸顔・横幅寄り**")
                else:
                    st.info("ℹ️ **判定目安: 標準的な卵型バランス**")

            st.divider()
            st.header("2. しわ・キメ診断エリア")
            col1_ana, col2_ana = st.columns(2)
            col1_ana.image(analyzed_a, caption="画像A: 分析エリア", use_container_width=True)
            col2_ana.image(analyzed_b, caption="画像B: 分析エリア", use_container_width=True)

            st.divider()
            st.header("3. 肌状態の数値化")
            
            metric_cols = st.columns(3)
            targets = ["おでこ", "ほうれい線周辺", "口元・あご周り"]
            
            for i, target in enumerate(targets):
                with metric_cols[i]:
                    st.subheader(f"■ {target}")
                    score_a = scores_a[target]
                    score_b = scores_b[target]
                    delta = score_b - score_a
                    
                    st.metric("画像A", f"{score_a:.1f}")
                    st.metric("画像B", f"{score_b:.1f}", delta=f"{delta:.1f}", delta_color="inverse")
            
            st.info("💡 ヒント: スコアが「0.0」になる場合は、サイドバーの「しわ検出感度」の数値を下げてみてください。")

        else:
            st.error("どちらかの画像から顔を検出できませんでした。")
elif file_a or file_b:
    st.info("比較のために、もう1枚の画像をアップロードしてください。")



