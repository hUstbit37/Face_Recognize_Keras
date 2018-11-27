# Face_Recognize_Keras
based FaceNet network was build by Google 2016
- môi trường chạy: python3.6 
- vào thư mục face-recognize-master
- cài đặt các lib bằng: pip3.6 install -r requirement.txt
- đầu tiên download image về bằng tool_download_image cho dữ liệu ảnh vào folder images
- run file dowload_landmarks.py
- tạo 1 thư mục image_crop để chửa ảnh sau khi detect_face
- run file dedect_face.py
- run file dump_model.py.Nếu không muốn mất thời gian training dữ liệu bạn có thể sử dựng trực tiếp model dã được train sẵn model.h5 trong weights
- run file predict_image.py. Trong file này có Path_Image là đường dẫn ảnh cần dự đoán
- run file camera_read.py để nhận dạng khuôn măt trực tiếp trong các frame mà camera bắn lên.

- ACC với bộ dữ liệu 10 class mỗi class xấp xỉ 100 image là: xấp xỉ 100% với data_training,và 88% với data_testing
