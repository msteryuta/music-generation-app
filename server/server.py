import socket
import os
import openai
from dotenv import load_dotenv
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio

load_dotenv()
api_key = os.getenv("API_KEY")
openai.api_key = api_key

if not api_key:
    raise ValueError("API Key error")


def start_server(host='0.0.0.0', port=12345):

    user_prompt = "indicate the melody and tone"

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        server_socket.bind((host, port))
        print(f"伺服器啟動，正在監聽 {host}:{port}")
        server_socket.listen(5)
        
        while True:
            print("等待客戶端連線...")
            client_socket, client_address = server_socket.accept()
            print(f"接收到來自 {client_address} 的連線")
            
            # 接收客戶端的訊息
            message = client_socket.recv(1024).decode('utf-8')
            print(f"收到訊息: {message}")
            
            # 生成音樂描述
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "only return instrument,Music style,key and rhythm"},
                    {"role": "user", "content": user_prompt + message}
                ],
                max_tokens=60
            )
            lyrics_description = completion['choices'][0]['message']['content']
            print("生成的音樂描述:", lyrics_description)

            # 加載模型並生成音頻
            processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small").to("cuda")

            inputs = processor(
                text=lyrics_description,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            audio_values = model.generate(**inputs, max_length=600)

            # 保存生成的音頻
            output_file = "generated_audio.wav"
            torchaudio.save(output_file, audio_values[0].cpu(), sample_rate=16000)
            print("生成的音頻已保存:", output_file)

            # 傳送音頻檔案至客戶端
            with open(output_file, "rb") as f:
                file_data = f.read()

            client_socket.sendall(file_data)
            print(f"音頻檔案 {output_file} 已發送至 {client_address}")

            # 關閉客戶端連線
            client_socket.close()
            print(f"已關閉 {client_address} 的連線\n")
    
    except KeyboardInterrupt:
        print("\n伺服器已關閉")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()
