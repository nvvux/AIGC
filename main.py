# main.py
"""
Chạy toàn bộ pipeline tối ưu hóa dịch vụ AIGC bằng Stackelberg Game học tăng cường.
- Huấn luyện tác tử ASP bằng TDRL
- Mô phỏng inference thực tế sau khi học xong
"""

from train import train_agent
from inference import run_realworld_inference

if __name__ == '__main__':
    print("🔧 Bắt đầu huấn luyện mô hình ASP với TDRL Agent...\n")
    trained_agent, reward_log = train_agent(epochs=1000)

    print("\n✅ Huấn luyện hoàn tất. Bắt đầu mô phỏng inference thực tế...\n")
    run_realworld_inference(trained_agent)
