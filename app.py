import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# ページ設定
st.set_page_config(page_title="顔診断アプリ (MediaPipe版)", layout="centered")

st.title("📸 顔バランス診断アプリ")
st.write("GoogleのMediaPipeを使って、顔のランドマーク（特徴点）を検出します。")

# ファイルアップロード
uploaded_file = st.file_uploader("顔写真をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を読み込む
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # MediaPipeの準備
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 顔検出の実行
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # 検出処理
        results = face_mesh.process(img_array)

        # 結果の描画
        if results.multi_face_landmarks:
            st.success("顔を認識しました！")
            
            # 元の画像をコピーして描画用にする
            annotated_image = img_array.copy()

            for face_landmarks in results.multi_face_landmarks:
                # 顔の網目（メッシュ）を描画
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # 輪郭などの線を描画
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # 画像を表示
            st.image(annotated_image, caption="解析結果", use_container_width=True)
            
            # データ活用のヒント（開発者用）
            st.info("💡 開発メモ: ここから目や鼻の座標を取り出して、黄金比などの計算に使えます。")
            
        else:
            st.error("顔が検出できませんでした。別の写真を試してみてください。")
