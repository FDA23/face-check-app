import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageOps

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

def draw_mesh(image, landmarks_proto):
    """顔のメッシュ（骨格）を描画する関数"""
    annotated_image = image.copy()
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
    return annotated_image

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

    mesh_img = draw_mesh(img_array, face_landmarks_proto)
    analyzed_img = img_array.copy()
    scores = {}

    # --- 解析エリアの定義（範囲拡大版）---
    
    # おでこ（変更なし）
    forehead_idx = [109, 338, 9, 336, 151]
    
    # ほうれい線エリア（頬骨の下から小鼻の横を含み、口角まで）
    nasolabial_idx = [
        205, 203, 36, 101, 50, 123, 117, 111, 147, 187, 207, # 左頬周辺
        425, 423, 266, 330, 280, 352, 346, 340, 376, 411, 427 # 右頬周辺
    ]
    
    # 口周り・マリオネットライン（口角の下、あご周辺まで広く）
    marionette_idx = [
        57, 186, 212, 287, 410, 432, 273, 335, 406, 313, 18, 83, 182, 106, 43 # 口周りとあご
    ]

    areas = {
        "おでこ": (forehead_idx, (0, 255, 0)),        # 緑
        "ほうれい線周辺": (nasolabial_idx, (255, 165, 0)), # オレンジ
        "口元・あご周り": (marionette_idx, (255, 0, 255))  # ピンク
    }

    for name, (indices, color) in areas.items():
        score, rect = analyze_area(img_array, face_landmarks_list, indices, name)
        scores[name] = score
        if rect:
            x, y, w, h = rect
            cv2.rectangle(analyzed_img, (x, y), (x+w, y+h), color, 2)

    return mesh_img, analyzed_img, scores

# --- メイン画面構成 ---
st.title("📸 顔バランス＆肌比較診断 (広範囲版)")
st.write("2枚の写真をアップロードして、骨格のゆがみと肌の状態（しわ・キメ）を比較します。")

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
            mesh_a, analyzed_a, scores_a = result_a
            mesh_b, analyzed_b, scores_b = result_b

            st.divider()
            st.header("1. 骨格・ゆがみの可視化")
            st.info("💡 チェックポイント: 正中のラインは真っ直ぐか？ 左右の目の高さは同じか？ 網目の形に注目してください。")
            col1_mesh, col2_mesh = st.columns(2)
            col1_mesh.image(mesh_a, caption="画像A: 骨格メッシュ", use_container_width=True)
            col2_mesh.image(mesh_b, caption="画像B: 骨格メッシュ", use_container_width=True)

            st.divider()
            st.header("2. しわ・キメ診断エリア（広範囲）")
            st.write("オレンジ（ほうれい線周辺）とピンク（口元・あご）の範囲を広げました。")
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
            
            st.info("💡 範囲を広げたため、以前より数値が大きくなっている可能性がありますが、比較（差分）には問題ありません。")

        else:
            st.error("どちらかの画像から顔を検出できませんでした。")
elif file_a or file_b:
    st.info("比較のために、もう1枚の画像をアップロードしてください。")
