from langchain_openai import OpenAI, ChatOpenAI

class LllmApi:
    def __init__(self, api_model_name, api_base_url, api_key, api_temperature, api_top_p, api_max_tokens, api_chat_mode=True):
        self.api_chat_mode = api_chat_mode
        
        if api_chat_mode:
            self.model = ChatOpenAI(
                model= api_model_name,
                base_url= api_base_url, 
                api_key= api_key,
                temperature= api_temperature, 
                top_p= api_top_p,
                max_tokens= api_max_tokens,
            )
        else:
            self.model = OpenAI(
                model= api_model_name,
                base_url= api_base_url, 
                api_key= api_key,
                temperature= api_temperature, 
                top_p= api_top_p,
                max_tokens= api_max_tokens,
            )

    def generate_answer(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if self.api_chat_mode:
            return self.model.invoke(messages).content
        return self.model.invoke(messages)
