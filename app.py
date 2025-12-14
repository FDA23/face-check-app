import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

# --- 設定 ---
st.set_page_config(page_title="顔バランス＆肌診断AI", layout="wide")
st.title("🪞 顔バランス＆肌診断AI（徹底比較モード）")

# --- AIモデルの準備 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- 関数群 ---
def get_landmark_point(landmarks, idx, w, h):
    point = landmarks[idx]
    return (int(point.x * w), int(point.y * h))

def process_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    h, w = img.shape[:2]
    aspect = h / w
    resize_w = 600
    resize_h = int(resize_w * aspect)
    img = cv2.resize(img, (resize_w, resize_h))
    
    analysis_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]

    # 1. 基準：黒目間の距離
    right_iris = get_landmark_point(lm, 473, w, h)
    left_iris = get_landmark_point(lm, 468, w, h)
    eye_dist = math.dist(right_iris, left_iris)
    
    # 2. 骨格計測
    nose_btm = get_landmark_point(lm, 2, w, h)
    lip_top = get_landmark_point(lm, 0, w, h)
    philtrum_ratio = math.dist(nose_btm, lip_top) / eye_dist
    
    eye_y = (right_iris[1] + left_iris[1]) / 2
    midface_ratio = abs(lip_top[1] - eye_y) / eye_dist
    
    # 3. 左右差計測
    nose_tip = get_landmark_point(lm, 4, w, h)
    center_x = nose_tip[0]
    cheek_r = get_landmark_point(lm, 234, w, h)
    cheek_l = get_landmark_point(lm, 454, w, h)
    width_r = abs(center_x - cheek_r[0])
    width_l = abs(center_x - cheek_l[0])
    ratio_total = (width_r + width_l) / eye_dist
    
    # 描画
    cv2.line(analysis_img, nose_btm, lip_top, (0, 255, 255), 3) # 黄色：人中
    cv2.line(analysis_img, (0, int(eye_y)), (w, int(eye_y)), (255, 255, 0), 1) # 水色：目のライン（参考）
    
    cv2.line(analysis_img, cheek_r, (center_x, cheek_r[1]), (0, 255, 0), 2)
    cv2.line(analysis_img, cheek_l, (center_x, cheek_l[1]), (0, 255, 0), 2)
    cv2.line(analysis_img, (center_x, 0), (center_x, h), (0, 0, 255), 1)

    wrinkle = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 4
    )
    wrinkle = cv2.medianBlur(wrinkle, 1)
    wrinkle_color = cv2.cvtColor(wrinkle, cv2.COLOR_GRAY2RGB)

    return {
        "name": image_file.name,
        "img_res": cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB),
        "img_wrinkle": wrinkle_color,
        "philtrum": philtrum_ratio,
        "midface": midface_ratio,
        "ratio_total": ratio_total
    }

def get_comment(diff, type_name):
    if abs(diff) < 0.005: return "変化なし（維持できています）"
    
    if type_name == "人中":
        if diff < 0: return "✨ 短くなりました（若見え効果！）"
        else: return "⚠️ 少し伸びています（たるみの可能性）"
    elif type_name == "中顔面":
        if diff < 0: return "✨ 引き締まりました（小顔効果！）"
        else: return "⚠️ 間延びしています（表情筋の衰え？）"
    elif type_name == "横幅":
        if diff < 0: return "✨ スッキリしました（むくみ解消！）"
        else: return "⚠️ 広がっています（エラ張り・むくみ？）"

# --- メイン画面 ---
uploaded_files = st.file_uploader("写真を2枚選んでください（比較用）", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    if st.button("比較診断スタート"):
        with st.spinner("AIが詳細レポートを作成中..."):
            res1 = process_image(uploaded_files[0])
            res2 = process_image(uploaded_files[1])
            
            if res1 and res2:
                st.success("分析完了！")
                
                # 画像エリア
                st.subheader("1. 骨格・ゆがみの可視化")
                c1, c2 = st.columns(2)
                c1.image(res1["img_res"], caption="画像1 (Before)")
                c2.image(res2["img_res"], caption="画像2 (After)")
                
                st.subheader("2. しわ・キメ診断")
                with st.expander("ℹ️ 肌診断画像の見方"):
                    st.markdown("""
                    * **黒い線・点** 👉 しわ、毛穴、キメの乱れ
                    * **黒い影** 👉 たるみによる影
                    * **比較方法** 👉 画像2で黒い部分が減っていればケア成功です！
                    """)
                c3, c4 = st.columns(2)
                c3.image(res1["img_wrinkle"], caption="画像1 肌状態")
                c4.image(res2["img_wrinkle"], caption="画像2 肌状態")

                # レポートエリア
                st.divider()
                st.subheader("3. 詳細比較レポート")
                
                # ★ここに追加しました：数値の定義説明
                with st.expander("❓ 数値の定義（どこの比率？）を見る"):
                    st.markdown("""
                    すべての数値は、**「左右の黒目の中心間の距離」を「1.0」とした場合の比率**です。
                    
                    * **人中（じんちゅう）比率**
                        * 「鼻の下」から「上唇の山」までの長さ。
                        * 短いほど若々しい印象になります。
                    * **中顔面（ちゅうがんめん）比率**
                        * 「目の高さ」から「上唇の山」までの長さ。
                        * 短いほど小顔・童顔に見えます。
                    * **顔の横幅比率**
                        * 鼻の中心から、左右のエラ（輪郭）までの合計幅。
                        * 小さいほどフェイスラインがスッキリしています。
                    """)

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    diff1 = res2['philtrum'] - res1['philtrum']
                    st.metric("人中比率", f"{res2['philtrum']:.3f}", f"{diff1:.3f}", delta_color="inverse")
                    st.info(get_comment(diff1, "人中"))

                with col_b:
                    diff2 = res2['midface'] - res1['midface']
                    st.metric("中顔面比率", f"{res2['midface']:.3f}", f"{diff2:.3f}", delta_color="inverse")
                    st.info(get_comment(diff2, "中顔面"))

                with col_c:
                    diff3 = res2['ratio_total'] - res1['ratio_total']
                    st.metric("顔の横幅比率", f"{res2['ratio_total']:.3f}", f"{diff3:.3f}", delta_color="inverse")
                    st.info(get_comment(diff3, "横幅"))

                st.divider()
            else:
                st.error("顔が見つかりませんでした。")
elif uploaded_files:
    st.warning("写真を2枚選択してください。")