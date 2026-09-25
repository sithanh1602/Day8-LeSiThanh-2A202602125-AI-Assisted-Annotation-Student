# Vì sao chọn lô này?

Nguồn: 50 dòng đầu của `outputs/selection_round1.csv`, đối chiếu ảnh trong `outputs/selection_round1.jpg`. Đây là phương án ưu tiên nếu chỉ đủ công rà năm ảnh; lô thực tế vẫn gồm 12 ảnh do notebook chọn.

## Năm ảnh ưu tiên

| Ưu tiên | Frame | Hạng CSV | Thời điểm (giây) | Score | U | A | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 72.8 | 0.9591 | 0.9182 | 1.0000 | Điểm cao nhất, 18 box mơ hồ; có cả xe đi tới và cụm đèn hậu cần rà. |
| 2 | frame_0369.jpg | 2 | 147.6 | 0.9324 | 0.9315 | 0.8889 | 16 box mơ hồ, nhiều xe/đèn trong ảnh; đáng kiểm tra nhưng công rà tương đối lớn. |
| 3 | frame_0326.jpg | 4 | 130.4 | 0.9155 | 0.9310 | 0.8333 | 15 box mơ hồ; đại diện đoạn thời gian khác, cần phân biệt thân xe với vùng đèn sáng. |
| 4 | frame_0099.jpg | 8 | 39.6 | 0.9063 | 0.9460 | 0.7778 | Bất định cao, 14 box mơ hồ; có xe xa/tối mà người học đã ghi nhận khi quan sát độc lập. |
| 5 | frame_0227.jpg | 11 | 90.8 | 0.8915 | 0.9164 | 0.7778 | Bổ sung đoạn thời gian khác, 14 box mơ hồ; cân bằng độ phủ thời gian và chi phí kiểm xe sát nhau. |

Tôi không lấy nguyên năm dòng đầu: `frame_0331.jpg` chỉ cách `frame_0326.jpg` 2 giây; `frame_0380.jpg` cách `frame_0369.jpg` 4.4 giây. Các ảnh này vẫn có thể hữu ích, nhưng trong ngân sách năm ảnh tôi ưu tiên thêm đoạn 39.6 và 90.8 giây. Khoảng cách thời gian chỉ là đại diện cho độ đa dạng, không bảo đảm các xe/cảnh hoàn toàn khác nhau. Phương án trên không phải kết quả một thí nghiệm train riêng.

## Ba ảnh model chọn và một ảnh cân nhắc khác

- `frame_0182.jpg`: `selected=True`, score 0.9591, đứng đầu. Contact sheet cho thấy hai chiều xe và cụm đèn hậu xa; cần kiểm từng xe thay vì chấp nhận toàn bộ nhãn AI.
- `frame_0369.jpg`: `selected=True`, score 0.9324. CSV có 43 dự đoán và 16 box mơ hồ; ảnh nhiều điểm sáng nên việc tách xe, loại phản chiếu và rà xe xa tốn công. Số dự đoán CSV không phải số xe thật hay số pre-label được giữ ở ngưỡng 0.25.
- `frame_0099.jpg`: `selected=True`, score 0.9063, U=0.9460. Quan sát độc lập ghi khoảng 23 xe và hai vùng khó ở phía trái; bản nhãn cuối có 25 box. Chênh lệch này cần đối chiếu ảnh, không tự chứng minh bản cuối đúng tuyệt đối.
- `frame_0372.jpg`: hạng 6, score 0.9101 nhưng `selected=False`. Ảnh ở 148.8 giây, chỉ cách ảnh đã chọn `frame_0369.jpg` 1.2 giây, không đạt `MIN_GAP_S=2.0`. Vì vậy điểm cao không đồng nghĩa được chọn. Trong ngân sách hạn chế, ưu tiên ảnh gần trùng có thể tốn công mà thêm ít thông tin.

## Giới hạn của phép chọn

Trong 50 ứng viên đầu, không có dòng `empty=True`. Ở vòng đầu D đều bằng 1 vì chưa có ảnh đã gán nhãn; do đó D không phân biệt thứ hạng giữa các ảnh này. Bộ lọc thời gian vẫn giúp hạn chế ảnh quá gần trong cùng lô.

Score ưu tiên sự không chắc chắn của model, không trực tiếp đo chất lượng nhãn hoặc mức tăng AP50 sau train. Xe bị bỏ sót hoàn toàn có thể không tạo box để tính bất định. Tôi cần kết hợp quan sát ảnh, công rà nhãn và mức trùng cảnh, thay vì chỉ chọn theo điểm. Không dùng ảnh test để chọn vào tập train.
