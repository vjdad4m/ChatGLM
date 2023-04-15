import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel

model_complete = 'THUDM/chatglm-6b'  # ~ 13 GB Memory
model_int4 = 'THUDM/chatglm-6b-int4' # ~ 5.2 GB Memory

tokenizer = AutoTokenizer.from_pretrained(model_int4, trust_remote_code=True)
model = AutoModel.from_pretrained(model_int4, trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

MSG = 'Welcome to the ChatGLM-6B model, enter the content to carry out the conversation, clear to clear the conversation history, stop to terminate the program.'

def build_prompt(history):
    prompt = MSG
    for query, response in history:
        prompt += f"\n\nUser: {query}"
        prompt += f"\n\nChatGLM-6B: {response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print(MSG)
    while True:
        query = input("\nUser: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print(MSG)
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
