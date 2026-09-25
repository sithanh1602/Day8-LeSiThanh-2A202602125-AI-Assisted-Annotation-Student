# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Sĩ Thành
Mssv: 2A202602125

Công cụ gán nhãn: CVAT Docker local; xuất Ultralytics YOLO Detection 1.0. Mô hình chạy trên Google Colab. Báo cáo dùng kết quả lần chạy cuối sau khi sửa lại nhãn xe bus, với 12 ảnh và 345 box; không dùng kết quả thử trước đó có 339 box.

## 1. Dữ liệu và cách chia tập

Dữ liệu là ảnh đường cao tốc ban đêm từ một camera cố định. Theo `data/DATA.md`, 400 frame được lấy ở 2.5 frame/giây; 268 ảnh thuộc pool, 20 ảnh thuộc test và 112 ảnh thuộc vùng đệm bị loại. Ảnh pool gần test nhất cách 4.4 giây.

Các frame gần nhau có thể chứa cùng một xe và nền gần như giống nhau. Nếu chia ngẫu nhiên, train và test có thể cùng chứa những xe đó, làm kết quả đánh giá lạc quan do rò rỉ dữ liệu. Chia theo thời gian và thêm vùng đệm giúp giảm nguy cơ này, nhưng chưa kiểm tra được khả năng tổng quát sang camera, tuyến đường hoặc điều kiện chiếu sáng khác.

Test có 417 box tham chiếu, trong đó 14 box cao dưới 16 pixel được bỏ qua; 403 box được dùng để đánh giá. Nhãn test do mô hình tạo và chưa được con người rà từng box. Các số đo dưới đây phản ánh mức khớp bộ tham chiếu này, không phải chất lượng thực địa tuyệt đối. Ảnh test không được đưa vào huấn luyện và không chỉnh nội dung box test.

## 2. Mô hình khởi đầu lạnh (cold start)

Model ban đầu là `yolov8n.pt` pretrained trên COCO, gộp car/bus/truck thành lớp `car` của lab. Mốc vòng 0 từ `reports/rounds_table.md`:

| Vòng | Ảnh train | Box train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Số chính xác trong `outputs/metrics_round0.json`: AP50=0.7714, precision=0.9249, recall=0.4888, F1=0.6396. Recall nhóm nhỏ chỉ 0.1818, thấp hơn nhóm vừa 0.5473 và lớn 0.5610. Model còn yếu với xe nhỏ/xa; ảnh so sánh cho thấy nhiều box vàng ở phía xa và các vùng thân xe tối quanh đèn hậu.

Ở `frame_0050.jpg`, cột cold start trong ảnh so sánh ghi TP=11, FP=2, FN=7. Cụm box đỏ quanh đèn ở vùng xa phía trên cần được đối chiếu lại ảnh gốc và nhãn tham chiếu trước khi kết luận chắc chắn là model phát hiện nhầm: có thể là xe khó thấy, phản chiếu, box lệch hoặc nhãn tham chiếu chưa đầy đủ. Đây là ca cần rà, không phải khẳng định tham chiếu sai. Việc rà chỉ ghi nhận trong báo cáo, không sửa test.

## 3. Chiến lược chọn mẫu

Giữ cấu hình bắt buộc `AL_K=12`, `STRATEGY="uncertainty"`, `MIN_GAP_S=2.0`. Điểm ảnh là `score = 0.5*U + 0.3*A + 0.2*D`:

- U: trung bình bất định của năm box khó nhất, với `u = 1 - |2*conf - 1|`; confidence gần 0.5 có bất định lớn.
- A: số box có confidence từ 0.15 đến dưới 0.5, chuẩn hóa theo giá trị lớn nhất trong pool.
- D: khoảng cách đến ảnh đã gán nhãn gần nhất theo thời gian, giới hạn 10 giây và chuẩn hóa. Trong vòng đầu, D=1 cho các ứng viên vì chưa có ảnh đã gán.

Notebook chọn điểm cao trước nhưng giữ khoảng cách tối thiểu hai giây giữa các ảnh được chọn. `reports/SELECTION.md` phân tích 50 ứng viên đầu và đề xuất năm ảnh ưu tiên khi hạn chế ngân sách.

Ba ví dụ được chọn là `frame_0182.jpg` (72.8 giây, score 0.9591, 18 box mơ hồ), `frame_0369.jpg` (147.6 giây, 0.9324, 16 box mơ hồ), và `frame_0099.jpg` (39.6 giây, 0.9063, 14 box mơ hồ). Contact sheet cho thấy cần rà cả xe đi tới, cụm đèn hậu và xe xa. Ngược lại, `frame_0372.jpg` có score 0.9101 nhưng chỉ cách `frame_0369.jpg` 1.2 giây nên bị loại bởi khoảng cách thời gian.

Điểm cao không chứng minh ảnh sẽ cải thiện model. Nhiều box mơ hồ có thể do ánh sáng hoặc dự đoán trùng; xe bị bỏ sót hoàn toàn có thể không đóng góp box vào điểm. Cần cân nhắc thời gian rà từng xe và tránh dùng nhiều ảnh gần trùng.

## 4. Các vòng học chủ động (active learning)

Đã hoàn thành một vòng bắt buộc. Nhãn cuối được sửa trên CVAT, đóng gói bằng `tools/pack_labels.py`, rồi fine-tune từ `yolov8n.pt` trong 50 epochs, kích thước đầu vào 960. Dùng `last.pt` để đánh giá trên cùng 20 ảnh test. File metrics cuối ghi runtime `cpu`; không dùng bảng mAP trên ảnh train làm số liệu test.

Bảng từ `reports/rounds_table.md` (làm tròn ba chữ số):

| Vòng | Model | Ảnh train | Box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 345 | 0.774 | +0.002 | 0.927 | 0.377 | 0.536 | 0.000 | 0.415 | 0.707 |

Theo `outputs/round1_diff.json`, AI đề xuất 169 box; nhãn cuối có 345 box: 155 accepted, 6 edited, 8 deleted và 184 added. Đây là thống kê ghép box bằng IoU (accepted từ 0.85, edited từ 0.50 đến dưới 0.85), không phải lịch sử thao tác CVAT. Một box bị kéo nhiều có thể được đếm thành xóa và thêm. Nhãn người sửa cũng có thể sai, vì vậy không coi mọi deleted là box giả chắc chắn hay mọi added là xe đúng chắc chắn.

`BLIND_SCAN.md` ghi khoảng 23 xe ở `frame_0099.jpg`, với hai vùng khó phía trên bên trái: xe chỉ thấy đèn phanh và xe gần đó nằm trong bóng tối. Quan sát đã được khóa trước khi nhập nhãn AI. Bản cuối có 25 box, so với 13 pre-label; thống kê ghi 12 accepted, 1 edited và 12 added. Quan sát nhanh và bản nhãn cuối là hai bằng chứng khác nhau; chênh lệch số lượng không tự chứng minh số nào đúng.

Người học nhận xét xe xa, có đèn phanh hoặc nằm trong vùng tối thường bị bỏ sót trong lô. Đây là nhận xét quá trình rà, không phải phép đo chứng minh tất cả xe dạng này đều bị bỏ sót. Ở `frame_0392.jpg`, người học ban đầu xóa khung xe bus vì nghĩ không thuộc lớp car, sau đó vẽ lại khi đối chiếu guideline: mọi xe từ bốn bánh trở lên đều thuộc car. Ca này cho thấy người rà cũng có thể hiểu sai quy tắc. Với xe tối chỉ thấy đèn, cần box theo phần thân có thể xác định, không chỉ khoanh chấm đèn hoặc ôm vệt phản chiếu; xe che khuất chỉ lấy phần nhìn thấy. Xe quá xa dưới khoảng 16 pixel cần xử lý nhất quán theo quy tắc bỏ qua.

Ở `frame_0380.jpg`, người học xác nhận đã tách một khung AI gộp hai xe ở xa thành hai khung riêng. Quyết định này tuân theo guideline: hai xe sát nhau vẫn là hai đối tượng, mỗi xe cần một box. Nhật ký `REVIEW_LOG.csv` ghi ca này cùng ca bổ sung xe xa ở `frame_0099.jpg` và khôi phục xe bus ở `frame_0392.jpg`. Diff toàn ảnh `frame_0380.jpg` ghi 2 edited và 17 added; không thể quy các số tổng này cho riêng thao tác tách khung nếu không có lịch sử đối tượng.

Kết quả chính xác từ `outputs/metrics_round1.json`: AP50=0.7737, tăng 0.0023 so với cold start và cũng là so với vòng trước. Precision tăng từ 0.9249 lên 0.9268, nhưng recall giảm từ 0.4888 xuống 0.3772 và F1 giảm từ 0.6396 xuống 0.5362. Tại conf=0.25, TP giảm 197 xuống 152, FP giảm 16 xuống 12 và FN tăng 206 lên 251.

Recall nhóm nhỏ giảm 0.1818 xuống 0; nhóm vừa giảm 0.5473 xuống 0.4155; nhóm lớn tăng 0.5610 lên 0.7073. AP50 tổng hợp theo các ngưỡng confidence nên có thể tăng nhẹ trong khi recall ở một ngưỡng cố định giảm. Không thể kết luận model tốt lên toàn diện từ AP50.

Trong `outputs/compare_round1.jpg`, `frame_0050.jpg` có xe lớn gần góc dưới bên trái chuyển từ box vàng (bỏ sót) ở cold start sang xanh lá sau train. Tuy nhiên tổng TP của ảnh giảm 11 xuống 10, FN tăng 7 lên 8, FP giảm 2 xuống 0: cải thiện ở một xe không đồng nghĩa cải thiện cả ảnh. `frame_0150.jpg` giảm TP từ 10 xuống 4 và tăng FN từ 10 lên 16, với nhiều xe xa/vừa còn box vàng. Giả thuyết là lô nhỏ hoặc thay đổi confidence khiến model ưu tiên xe lớn hơn; cần kiểm phân bố kích thước nhãn và confidence để xác nhận, chưa đủ bằng chứng quy toàn bộ thay đổi cho việc sửa một xe bus.

## 5. Kết luận và giới hạn

Đề xuất dừng sau vòng bắt buộc để rà chất lượng nhãn và phân tích kết quả, chưa train thêm ngay. AP50 tăng 0.0023, dưới mức 0.01 được lưu ý trong `data/DATA.md`, trong khi recall/F1 giảm và nhóm xe nhỏ vẫn yếu. Chưa có cơ sở kết luận cải thiện đáng tin cậy hoặc tổng quát.

Nếu tiếp tục, ưu tiên hai loại ca từ pool chưa gán nhãn: (1) xe xa có đèn hậu nhưng thân tối, còn đủ kích thước để đánh giá; (2) xe cỡ vừa bị che một phần hoặc sát xe khác trong vùng tối. Ca đầu tốn công phóng to/phân biệt đèn phản chiếu; ca sau tốn công xác định phần nhìn thấy và tách box. Hạn chế lấy nhiều frame liên tiếp của cùng xe, kết hợp điểm bất định với kiểm tra ảnh và khoảng cách thời gian. Các frame test chỉ là ví dụ chẩn đoán, không lấy vào train vòng sau.

Trước khi train tiếp, cần kiểm box trùng, thân xe và vệt sáng, quy tắc xe bus/truck, tọa độ/class 0, số ảnh/box đóng gói, và sự nhất quán với xe rất nhỏ. Đối chiếu nhãn cuối, nhật ký và diff; xem riêng nhóm xe nhỏ/vừa bị giảm recall. Không tăng số box chỉ để kỳ vọng điểm tăng và không chỉnh test hoặc số đo.

Giới hạn chính: 12 ảnh train từ một cảnh cố định, test chỉ 20 ảnh có tương quan thời gian, nhãn tham chiếu do model tạo chưa được người rà, và phép chấm bỏ qua một số xe quá nhỏ. Chưa có kiểm định thống kê hoặc đánh giá trên camera khác. Nhãn đã sửa là dữ liệu huấn luyện do người rà, còn chất lượng model sau train phải được xem qua số đo test và ảnh so sánh; hai loại chất lượng này không đồng nhất.

Tự kiểm: nhãn cuối có 12 ảnh/345 box, khớp metrics vòng 1; đã lưu ảnh so sánh và bảng vòng; không đưa ảnh test vào train. Nhật ký ghi ba ca thực tế do người học xác nhận. Báo cáo và phân tích lựa chọn được hỗ trợ soạn bằng AI từ kết quả thực nghiệm, quan sát người học cung cấp và file trong repo; người học cần đọc lại nội dung trước khi nộp.
